#include <matmul/matmul_mma_m2n2k16.h>
#include <matmul/matmul_mma_m2n4k16.h>

#include <cstdlib>

#include "mbed.h"

using namespace mculib;


// #define DEBUG 1

Timer t;

#ifdef DEBUG
void matmul_golden(int8_t* ptr_A, int8_t* ptr_B, int8_t* ptr_C, float* scales, int8_t* ptr_bias,
                   int M, int N, int K, int32_t B_offset, int32_t C_offset, int32_t clip_min,
                   int32_t clip_max) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int32_t acc = ptr_bias[i];
      for (int k = 0; k < K; ++k) {
        acc += ptr_A[i * K + k] * ((int32_t)ptr_B[k * N + j] + B_offset);
      }
      acc = (int32_t)((float)(acc)*scales[i]) + C_offset;
      acc = MAX(clip_min, MIN(clip_max, acc));
      ptr_C[i * N + j] = (int8_t)(acc);
    }
  }
}
#endif

#define M 128
#define N 512
#define K 512

const int8_t A[M * K] = {0};
int8_t B[K * N] = {0};
int8_t C[M * N] = {0};
float scales[M] = {0};
int8_t bias[M] = {0};

#ifdef DEBUG
int8_t A_data[M * K] = {0};
int8_t B_data[K * N] = {0};
int8_t C_data[M * N] = {0};
#endif

int main() {
  // init parameters
  int32_t B_offset = 4343;
  int32_t C_offset = 893;
  int32_t output_clip_min = -9999;
  int32_t output_clip_max = 9999;

  // init data
  //   for (int i = 0; i < M; ++i) {
  //     for (int k = 0; k < K; ++k) {
  //       A[i * K + k] = rand() % 8;
  // #ifdef DEBUG
  //       A_data[i * K + k] = A[i * K + k];
  // #endif
  //     }
  //   }

  for (int j = 0; j < N; ++j) {
    for (int k = 0; k < K; ++k) {
      B[k * N + j] = rand() % 8;
#ifdef DEBUG
      B_data[k * N + j] = B[k * N + j];
#endif
    }
  }

  for (int i = 0; i < M; ++i) {
    scales[i] = 1.0;
    bias[i] = rand() % 8;
  }

  t.start();

  matmul_s8s8s8_acc32_mx_n4x_k16x_row_row_mma_m2n2k16_boffset(&A[0], &B[0], &C[0], &scales[0],
                                                              &bias[0], M, N, K, B_offset, C_offset,
                                                              output_clip_min, output_clip_max);

  t.stop();
  printf("The time taken was %llu milliseconds\n",
         std::chrono::duration_cast<std::chrono::milliseconds>(t.elapsed_time()).count());

#ifdef DEBUG
  matmul_golden(&A_data[0], &B_data[0], &C_data[0], &scales[0], &bias[0], M, N, K, B_offset,
                C_offset, output_clip_min, output_clip_max);
#endif

  printf("End of execution\n");

#ifdef DEBUG
  int errors = 0;
  for (int i = 0; i < M * N; ++i) {
    if (C[i] != C_data[i]) {
      errors += 1;
    }
  }
  if (errors == 0) {
    printf("Correctness check passed!\n");
  } else {
    printf("Errors (%d)!\n", errors);
  }
#endif
  return 0;
}