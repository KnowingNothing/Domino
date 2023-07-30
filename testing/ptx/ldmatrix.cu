#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cassert>
#include <iostream>

__global__ void warp_ld_matrix(half* A, half* B) {
  half regv[8];

  __shared__ half smem[16][16];
  int tx = threadIdx.x;
  for (int i = 0; i < 8; ++i) {
    smem[tx / 2][tx % 2 * 8 + i] = A[tx / 2 * 16 + tx % 2 * 8 + i];
  }
  __syncthreads();

  int grp_m = tx / 16 % 2;
  int grp_n = tx / 8 % 2;
  int grp_local_idx = tx % 8;
  uint32_t src_ptr;
  void* ptr = (void*)(&smem[grp_m * 8 + grp_local_idx][grp_n * 8]);

  asm volatile("{ .reg .u64 tmptr; cvta.to.shared.u64 tmptr, %1; cvt.u32.u64 %0, tmptr; }\n"
               : "=r"(src_ptr)
               : "l"(ptr));

  uint32_t* dst = (uint32_t*)&regv[0];
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
               : "r"(src_ptr));

  for (int i = 0; i < 8; ++i) {
    B[(tx / 4 + i / 4 * 8) * 16 + tx % 4 * 2 + i / 2 % 2 * 8 + i % 2] = regv[i];
  }
}

int main() {
  half* hA = (half*)malloc(16 * 16 * 2);
  half* hB = (half*)malloc(16 * 16 * 2);

  for (int i = 0; i < 16 * 16; ++i) {
    hA[i] = (half)(rand() % 7 * 1377.999);
  }

  half *dA, *dB;
  cudaMalloc(&dA, 16 * 16 * 2);
  cudaMalloc(&dB, 16 * 16 * 2);

  cudaMemcpy(dA, hA, 16 * 16 * 2, cudaMemcpyHostToDevice);

  dim3 blocks(1, 1, 1);
  dim3 threads(32, 1, 1);
  warp_ld_matrix<<<blocks, threads>>>(dA, dB);

  cudaMemcpy(hB, dB, 16 * 16 * 2, cudaMemcpyDeviceToHost);

  //   for (int i = 0; i < 16; ++i) {
  //     for (int j = 0; j < 16; ++j) {
  //         std::cout << (float)hB[i * 16 + j] << " ";
  //     }
  //     std::cout << "\n";
  //   }

  int errors = 0;
  for (int i = 0; i < 16 * 16; ++i) {
    if ((float)hA[i] != (float)hB[i]) {
      errors += 1;
      // std::cout << (float)hA[i] - (float)hB[i] << "\n";
    }
  }

  assert(errors == 0);
  std::cout << "Correctness Passed!\n";

  free(hA);
  free(hB);
  cudaFree(dA);
  cudaFree(dB);
  return 0;
}