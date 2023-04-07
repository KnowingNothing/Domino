from ..base import TileFlowAcceleratorBase
from ..base import HardwareLevel
from .buffer import Buffer
from .pe import Engine, ALU


def tileflow_accelerator_generator(acc: TileFlowAcceleratorBase):
    assert isinstance(acc, TileFlowAcceleratorBase), type(acc)

    indent_ = 0

    def make_indent():
        ret = ""
        for i in range(indent_):
            ret += "  "
        return ret

    def increase_indent():
        nonlocal indent_
        indent_ += 1

    def decrease_indent():
        nonlocal indent_
        indent_ -= 1

    def walk_on_buffer(buf: Buffer):
        assert isinstance(buf, Buffer)
        ind = make_indent()
        ret = f"name: {buf.name}"
        if buf.instance > 1:
            ret += f"[0..{buf.instance-1}]"
        ret += "\n"
        ret += f"{ind}class: {buf.buffer_class}\n"
        ret += f"{ind}attributes:\n"
        attr_names = [
            "width", "depth", "meshX", "meshY",
            "word_bits", "data_width", "block_size",
            "technology", "dram_type",
            "read_bandwidth", "write_bandwidth",
            "bandwidth", "cluster_size"]
        translate_keys = {
            "width": "width",
            "depth": "depth",
            "meshX": "meshX",
            "meshY": "meshY",
            "word_bits": "word-bits",
            "data_width": "data-width",
            "block_size": "block-size",
            "technology": "technology",
            "dram_type": "type",
            "read_bandwidth": "read_bandwidth",
            "write_bandwidth": "write_bandwidth",
            "bandwidth": "bandwidth",
            "cluster_size": "cluster-size"
        }
        for attr_name in attr_names:
            if hasattr(buf, attr_name) and getattr(buf, attr_name) is not None:
                value = getattr(buf, attr_name)
                key_name = translate_keys[attr_name]
                ret += f"{ind}  {key_name}: {value}\n"

        if len(buf.sub_levels):
            sub_level_results = []
            increase_indent()
            for sub in buf.sub_levels:
                sub_level_results.append(walker(sub))
            decrease_indent()
            ret += f"{ind}subtree:\n"
            for sub in sub_level_results:
                ret += f"{ind}- "
                ret += sub
        return ret

    def walk_on_alu(alu: ALU):
        assert isinstance(alu, ALU)
        ind = make_indent()
        ret = f"name: {alu.name}"
        if alu.instance > 1:
            ret += f"[0..{alu.instance-1}]"
        ret += "\n"
        ret += f"{ind}class: {alu.alu_class}\n"
        ret += f"{ind}attributes:\n"
        attr_names = [
            "meshX", "meshY",
            "data_width",
            "technology"]
        translate_keys = {
            "meshX": "meshX",
            "meshY": "meshY",
            "data_width": "data-width",
            "technology": "technology"
        }
        for attr_name in attr_names:
            if hasattr(alu, attr_name) and getattr(alu, attr_name) is not None:
                value = getattr(alu, attr_name)
                key_name = translate_keys[attr_name]
                ret += f"{ind}  {key_name}: {value}\n"

        if len(alu.sub_levels):
            sub_level_results = []
            increase_indent()
            for sub in alu.sub_levels:
                sub_level_results.append(walker(sub))
            decrease_indent()
            ret += f"{ind}subtree:\n"
            for sub in sub_level_results:
                ret += f"{ind}- "
                ret += sub
        return ret

    def walk_on_engine(engine: Engine):
        assert isinstance(engine, Engine)
        ind = make_indent()
        ret = f"name: {engine.name}"
        if engine.instance > 1:
            ret += f"[0..{engine.instance-1}]"
        ret += "\n"
        ret += f"{ind}attributes:\n"
        attr_names = ["technology"]
        translate_keys = {
            "technology": "technology"
        }
        for attr_name in attr_names:
            if hasattr(engine, attr_name) and getattr(engine, attr_name) is not None:
                value = getattr(engine, attr_name)
                key_name = translate_keys[attr_name]
                ret += f"{ind}  {key_name}: {value}\n"

        if len(engine.local):
            local_results = []
            increase_indent()
            for l in engine.local:
                local_results.append(walker(l))
            decrease_indent()
            ret += f"{ind}local:\n"
            for l in local_results:
                ret += f"{ind}- "
                ret += l

        if len(engine.sub_levels):
            sub_level_results = []
            increase_indent()
            for sub in engine.sub_levels:
                sub_level_results.append(walker(sub))
            decrease_indent()
            ret += f"{ind}subtree:\n"
            for sub in sub_level_results:
                ret += f"{ind}- "
                ret += sub
        return ret

    def walker(level: HardwareLevel):
        assert isinstance(level, HardwareLevel), type(level)
        vtable = {
            Buffer: walk_on_buffer,
            ALU: walk_on_alu,
            Engine: walk_on_engine
        }
        func = vtable[level.__class__]
        return func(level)

    result = "architecture:\n"
    increase_indent()
    result += f"  version: {acc.version}\n"
    if acc.hardware_level is not None:
        result += "  subtree:\n"
        increase_indent()
        hw = walker(acc.hardware_level)
        decrease_indent()
        result += f"  - {hw}"
    decrease_indent()
    return result
