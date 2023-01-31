from domino.type_system import DType
from domino.program_ir import Kernel

import serial
import numpy as np

from dataclasses import dataclass
from typing import List, Dict, Union
from tempfile import mkdtemp
import os
import shutil
import sys
import math


@dataclass
class BufferInfo:
    name: str
    dtype: DType
    offset: int
    nbytes: int
    const: bool = False

    def range_slice(self):
        return slice(self.offset, self.offset + self.nbytes)


@dataclass
class KernelInfo:
    name: str
    kname: str
    decl: str
    code: str

    @staticmethod
    def make(name, kernel: Kernel):
        sig = kernel.gen_signature()
        return KernelInfo(
            name=name,
            kname=kernel.signature.kernel_name,
            decl=sig,
            code=f"{sig} {{\n{kernel.source}}}",
        )


@dataclass
class CallInfo:
    name: str
    args: List[Union[str, int, float]]


@dataclass
class InvokeInfo:
    name: str
    calls: List[CallInfo]
    index: int = -1


class MbedRuntime:

    _COMMON_HEADER = """
#include <matmul/matmul_mma_m2n2k16.h>
#include <matmul/matmul_mma_m2n4k16.h>

#include <cstdlib>
#include <cmath>

#include "mbed.h"

using namespace mculib;

typedef int8_t int8;
typedef int16_t int16;
typedef int32_t int32;
typedef float float32;
"""

    def __init__(
        self,
        serial_port,
        target_name,
        mount_point,
        toolchain="GCC_ARM",
        timeout=None,
        cache_work_dir=True,
        other_serial_configs=None,
    ) -> None:
        if other_serial_configs is None:
            other_serial_configs = dict()
        print(f"changing permission of {serial_port} ...", file=sys.stderr)
        os.system(f"sudo chmod o+rw {serial_port}")
        self._serial = serial.Serial(
            serial_port, timeout=timeout, **other_serial_configs
        )
        assert os.path.exists(mount_point), f"mount point {mount_point} does not exist"
        self._target_name = target_name
        self._mount_point = mount_point
        self._toolchain = toolchain

        self._data_arena = bytearray()
        self._const_arena = bytearray()

        self._remote_data_arena_size = 0
        self._remote_data_dirty = False
        self._local_data_dirty = True
        self._local_const_dirty = True

        self._buffers: Dict[str, BufferInfo] = dict()
        self._kernels: Dict[str, KernelInfo] = dict()
        self._invokes: Dict[str, InvokeInfo] = dict()

        self._cache_work_dir = cache_work_dir
        self._work_dir = (
            os.getenv(
                "DOMINO_MBED_RT_WORK_DIR",
                str(os.path.join(os.getenv("HOME"), ".local/.domino_mbed_runtime")),
            )
            if cache_work_dir
            else mkdtemp(prefix="domino_mbed_runtime")
        )
        if not (cache_work_dir and os.path.exists(self._work_dir)):
            print(f"creating mbed workspace in {self._work_dir} ...", file=sys.stderr)
            os.system(f"mbed_tools new -c {self._work_dir}")
            os.symlink(
                self._get_mbed_os_path(),
                os.path.join(self._work_dir, "mbed-os"),
                target_is_directory=True,
            )

    def __del__(self):
        self._serial.close()
        if not self._cache_work_dir:
            os.unlink(os.path.join(self._work_dir, "mbed-os"))
            shutil.rmtree(self._work_dir, ignore_errors=True)

    @staticmethod
    def _get_mbed_os_path():
        path = os.getenv(
            "MBED_OS_PATH", str(os.path.join(os.getenv("HOME"), ".local/mbed-os"))
        )
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print(f"downloading mbed-os to {path} ...", file=sys.stderr)
            os.system(f"git clone git@github.com:ARMmbed/mbed-os.git --depth=1 {path}")
        return path

    @staticmethod
    def _u32_to_bytes(v):
        return bytes([v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0xFF])

    def _write_to_remote(self, offset, nbytes, data: bytes):
        self._serial.write(
            b"".join(
                [
                    b"W",
                    self._u32_to_bytes(offset),
                    self._u32_to_bytes(nbytes),
                    data,
                ]
            )
        )
        ack = self._serial.read_until(b"\n")
        assert b"done" in ack, ack

    def _read_from_remote(self, offset, nbytes):
        self._serial.write(
            b"".join(
                [
                    b"R",
                    self._u32_to_bytes(offset),
                    self._u32_to_bytes(nbytes),
                ]
            )
        )
        data = self._serial.read(nbytes)
        ack = self._serial.read_until(b"\n")
        assert b"done" in ack, ack
        return data

    def _push_data(self):
        size = min(len(self._data_arena), self._remote_data_arena_size)
        print(f"pushing {size} bytes data ...", file=sys.stderr)
        self._write_to_remote(0, size, self._data_arena[:size])
        self._local_data_dirty = False
        print("pushing data done", file=sys.stderr)

    def _pull_data(self):
        size = min(len(self._data_arena), self._remote_data_arena_size)
        print(f"pulling {size} bytes data ...", file=sys.stderr)
        data = self._read_from_remote(0, size)
        self._data_arena[:size] = data
        self._remote_data_dirty = False
        print("pulling data done", file=sys.stderr)

    def _sync_remote(self):
        if self._remote_data_dirty:
            self._pull_data()

    def _sync_local(self):
        if self._local_const_dirty:
            self._compile()
        if self._local_data_dirty:
            self._push_data()

    def alloc_buffer(
        self, name, nbytes, dtype: DType, offset=None, const=False, data=None
    ):
        if not isinstance(dtype, DType):
            dtype = DType.from_string(str(dtype))
        dt_bytes = dtype.bits // 8
        arena = self._const_arena if const else self._data_arena
        if offset is None:
            offset = int(math.ceil(len(arena) / dt_bytes)) * dt_bytes
        assert offset % dt_bytes == 0 and nbytes % dt_bytes == 0
        buf = BufferInfo(name, dtype, offset, nbytes, const)
        self._buffers[name] = buf
        self._local_const_dirty = True

        rest = offset + nbytes - len(arena)
        if rest > 0:
            arena.extend(b"\x00" * rest)
        if data is not None:
            self.set_buffer_data(name, data)

    def alloc_buffer_from_numpy(self, name, arr: np.ndarray, offset=None, const=False):
        data = arr.tobytes()
        self.alloc_buffer(name, len(data), arr.dtype, offset, const, data)

    def set_buffer_data(self, name, data: bytes):
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        buf = self._buffers[name]
        assert buf.nbytes == len(data)
        if buf.const:
            self._const_arena[buf.range_slice()] = data
            self._local_const_dirty = True
        else:
            self._sync_remote()
            self._data_arena[buf.range_slice()] = data
            self._local_data_dirty = True

    def get_buffer_data(self, name):
        buf = self._buffers[name]
        if buf.const:
            return self._const_arena[buf.range_slice()]
        else:
            self._sync_remote()
            return self._data_arena[buf.range_slice()]

    def get_buffer_numpy(self, name):
        buf = self._buffers[name]
        if buf.const:
            data = self._const_arena[buf.range_slice()]
        else:
            self._sync_remote()
            data = self._data_arena[buf.range_slice()]
        return np.frombuffer(data, dtype=str(buf.dtype))

    def set_kernel(self, name, kernel: Kernel):
        kinfo = KernelInfo.make(name, kernel)
        self._kernels[name] = kinfo
        self._gen_kernel_file(kinfo)
        self._local_const_dirty = True

    def set_invoke(self, name, calls: List[CallInfo]):
        calls = [ivk if isinstance(ivk, CallInfo) else CallInfo(*ivk) for ivk in calls]
        ivk = self._invokes.get(name, InvokeInfo(name, calls, len(self._invokes)))
        ivk.calls = calls
        self._invokes[name] = ivk
        self._local_const_dirty = True

    def execute(self, name):
        self._sync_local()
        print(f"executing '{name}' ...", file=sys.stderr)
        self._serial.write(
            b"".join(
                [
                    b"E",
                    self._u32_to_bytes(self._invokes[name].index),
                ]
            )
        )
        ack = self._serial.read_until(b"\n")
        assert b"done" in ack, ack
        self._remote_data_dirty = True
        print("executing done", file=sys.stderr)

    def execute_and_time(self, name, number=1, repeat=1):
        self._sync_local()
        self._serial.write(
            b"".join(
                [
                    b"T",
                    self._u32_to_bytes(self._invokes[name].index),
                ]
            )
        )
        # self._serial.write(self._u32_to_bytes(number))
        # self._serial.write(self._u32_to_bytes(repeat))
        ack = self._serial.read_until(b"\n")
        assert b"done" in ack, ack
        self._remote_data_dirty = True
        return ack.decode()

    def _compile(self):
        self._sync_remote()

        self._gen_cmake_file()
        self._gen_main_file()

        old_dir = os.getcwd()
        os.chdir(self._work_dir)

        print("compiling ...", file=sys.stderr)
        assert not os.system(
            f"mbed_tools compile -t {self._toolchain} -m {self._target_name}"
        ), "compiling failed"
        print("compiling done", file=sys.stderr)
        
        print("flashing ...", file=sys.stderr)
        assert not os.system(
            rf"sudo cp $(find -regex '.*{self._target_name}.*domino_mbed_runtime\.bin') {self._mount_point}/"
        ), "flashing failed"
        old_timeout = self._serial.timeout
        self._serial.timeout = 0.1
        for _ in range(100):
            self._serial.write(b"H")
            ack = self._serial.read_until(b"\n")
            if b"hello" in ack:
                break
        assert b"hello" in ack, ack
        self._serial.timeout = old_timeout
        print("flashing done", file=sys.stderr)

        os.chdir(old_dir)

        self._local_data_dirty = True
        self._local_const_dirty = False
        self._remote_data_arena_size = len(self._data_arena)

    def _gen_kernel_file(self, kinfo: KernelInfo):
        code = f"{self._COMMON_HEADER}\n{kinfo.code}"
        with open(os.path.join(self._work_dir, f"{kinfo.name}.cpp"), "w") as wf:
            wf.write(code)

    def _gen_cmake_file(self):
        code = f"""
cmake_minimum_required(VERSION 3.19.0)

set(MBED_PATH ${{CMAKE_CURRENT_SOURCE_DIR}}/mbed-os CACHE INTERNAL "")
set(MBED_CONFIG_PATH ${{CMAKE_CURRENT_BINARY_DIR}} CACHE INTERNAL "")
set(APP_TARGET domino_mbed_runtime)

include(${{MBED_PATH}}/tools/cmake/app.cmake)

project(${{APP_TARGET}})

add_subdirectory(${{MBED_PATH}})

add_executable(${{APP_TARGET}}
    main.cpp {' '.join(name + '.cpp' for name in self._kernels)}
)

target_link_libraries(${{APP_TARGET}} mbed-baremetal)

mbed_set_post_build(${{APP_TARGET}})

option(VERBOSE_BUILD "Have a verbose build process")
if(VERBOSE_BUILD)
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif()
        """
        with open(os.path.join(self._work_dir, "CMakeLists.txt"), "w") as wf:
            wf.write(code)

    def _gen_main_file(self):
        id2ivk = sorted(list(self._invokes.values()), key=lambda ivk: ivk.index)
        ivk_str = ""
        for ind, ivk in enumerate(id2ivk):
            calls_str = " ".join(
                f'{self._kernels[call.name].kname}({", ".join(str(a) for a in call.args)});'
                for call in ivk.calls
            )
            ivk_str += f"void invoke{ind}() {{ {calls_str} }}\n"
        ivk_str += f"void (* invokes[{len(id2ivk)}])() = {{ {', '.join(f'invoke{i}' for i in range(len(id2ivk)))} }};\n"

        code = (
            f"""        
{self._COMMON_HEADER}

{' '.join(k.decl + ';' for k in self._kernels.values())}

alignas(int64_t) uint8_t data_arena[{len(self._data_arena)}];
alignas(int64_t) const uint8_t const_arena[{len(self._const_arena)}] = {{{','.join(str(b) for b in self._const_arena)}}};

{' '.join(
    f'const {buf.dtype}* {buf.name} = (const {buf.dtype}*)(const_arena + {buf.offset});'
    if buf.const
    else f'{buf.dtype}* {buf.name} = ({buf.dtype}*)(data_arena + {buf.offset});'
    for buf in self._buffers.values()
)}

{ivk_str}
        """
            + r"""
uint32_t get_u32() {
  uint32_t v0 = fgetc(stdin) & 0xff;
  uint32_t v1 = fgetc(stdin) & 0xff;
  uint32_t v2 = fgetc(stdin) & 0xff;
  uint32_t v3 = fgetc(stdin) & 0xff;
  return (v3 << 24) | (v2 << 16) | (v1 << 8) | v0;
}

void exec_it() {
  uint32_t idx = get_u32();
  invokes[idx]();
  printf("done\n");
}

void time_it() {
  uint32_t idx = get_u32();
  void (* ivk)() = invokes[idx];
  Timer timer;
  timer.start();
  ivk();
  timer.stop();
  printf("done: %lld ms\n",
         std::chrono::duration_cast<std::chrono::milliseconds>(timer.elapsed_time()).count());
}

void read_data() {
  uint32_t base = get_u32();
  uint32_t size = get_u32();
  fwrite(data_arena + base, sizeof(int8_t), size, stdout);
  printf("done\n");
}

void write_data() {
  uint32_t base = get_u32();
  uint32_t size = get_u32();
  fread(data_arena + base, sizeof(int8_t), size, stdin);
  printf("done\n");
}

int main() {
  while (1) {
    int act = fgetc(stdin);
    switch (act) {
      case 'H':
        printf("hello\n");
        break;
      case 'E':
        exec_it();
        break;
      case 'T':
        time_it();
        break;
      case 'R':
        read_data();
        break;
      case 'W': 
        write_data();
        break;
      default:
        printf("unknown action: %c\n", act);
        break;
    }
  }
}
        """
        )
        with open(os.path.join(self._work_dir, "main.cpp"), "w") as wf:
            wf.write(code)
