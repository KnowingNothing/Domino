#ifndef MCULIB_MATMUL_MATMUL_MMA_M2N4K16_H
#define MCULIB_MATMUL_MATMUL_MMA_M2N4K16_H

#include <mma/mma_n2k4x.h>
#include <pack_value.h>

namespace mculib {

void matmul_s8s8s8_acc32_mx_n4x_k16x_row_row_mma_m2n2k16_boffset(
    const int8_t* ptr_A, int8_t* ptr_B, int8_t* ptr_C, float* scales, int8_t* ptr_bias, int M, int N,
    int K, int32_t B_offset, int32_t C_offset, int32_t clip_min, int32_t clip_max) {
  int remain_M = M & 1;
  int new_M = M - remain_M;

  const int16_t inoff16 = B_offset;
  int32_t offset_q15x2 = PACKs16x2(inoff16, inoff16);

  for (int mo = 0; mo < new_M; mo += 2) {
    float scale0 = scales[mo], scale1 = scales[mo + 1];
    for (int no = 0; no < N; no += 4) {
      int32_t acc00 = ptr_bias[mo], acc01 = ptr_bias[mo], acc02 = ptr_bias[mo],
              acc03 = ptr_bias[mo], acc10 = ptr_bias[mo + 1], acc11 = ptr_bias[mo + 1],
              acc12 = ptr_bias[mo + 1], acc13 = ptr_bias[mo + 1];
      for (int ko = 0; ko < K; ko += 16) {
        int8_t frag_B[4][16] = {0};
        for (int ki = 0; ki < 16; ++ki) {
          int32_t tmp;
          memcpy(&tmp, ptr_B + (ko + ki) * N + no, 4);
          frag_B[0][ki] = tmp & 0xff;
          frag_B[1][ki] = (tmp >> 8) & 0xff;
          frag_B[2][ki] = (tmp >> 16) & 0xff;
          frag_B[3][ki] = (tmp >> 24) & 0xff;
        }

        const int8_t* A0 = ptr_A + mo * K + ko;
        const int8_t* A1 = A0 + K;
        mma_m2n2k16_s8s8s8_acc32_boffset(A0, &frag_B[0][0], A1, &frag_B[1][0], B_offset, acc00,
                                         acc01, acc10, acc11);
        mma_m2n2k16_s8s8s8_acc32_boffset(A0, &frag_B[2][0], A1, &frag_B[3][0], B_offset, acc02,
                                         acc03, acc12, acc13);
      }
      int8_t* output00 = ptr_C + mo * N + no;
      int8_t* output01 = output00 + 1;
      int8_t* output02 = output01 + 1;
      int8_t* output03 = output02 + 1;
      int8_t* output10 = output00 + N;
      int8_t* output11 = output10 + 1;
      int8_t* output12 = output11 + 1;
      int8_t* output13 = output12 + 1;

      acc00 = (int32_t)((float)(acc00)*scale0) + C_offset;
      acc01 = (int32_t)((float)(acc01)*scale0) + C_offset;
      acc02 = (int32_t)((float)(acc02)*scale0) + C_offset;
      acc03 = (int32_t)((float)(acc03)*scale0) + C_offset;
      acc10 = (int32_t)((float)(acc10)*scale1) + C_offset;
      acc11 = (int32_t)((float)(acc11)*scale1) + C_offset;
      acc12 = (int32_t)((float)(acc12)*scale1) + C_offset;
      acc13 = (int32_t)((float)(acc13)*scale1) + C_offset;

      acc00 = MIN(clip_max, MAX(clip_min, acc00));
      acc01 = MIN(clip_max, MAX(clip_min, acc01));
      acc02 = MIN(clip_max, MAX(clip_min, acc02));
      acc03 = MIN(clip_max, MAX(clip_min, acc03));
      acc10 = MIN(clip_max, MAX(clip_min, acc10));
      acc11 = MIN(clip_max, MAX(clip_min, acc11));
      acc12 = MIN(clip_max, MAX(clip_min, acc12));
      acc13 = MIN(clip_max, MAX(clip_min, acc13));

      *(output00) = (int8_t)acc00;
      *(output01) = (int8_t)acc01;
      *(output02) = (int8_t)acc02;
      *(output03) = (int8_t)acc03;
      *(output10) = (int8_t)acc10;
      *(output11) = (int8_t)acc11;
      *(output12) = (int8_t)acc12;
      *(output13) = (int8_t)acc13;
    }
  }

  if (remain_M) {
    float scale0 = scales[new_M];
    for (int no = 0; no < N; no += 4) {
      int32_t acc00 = ptr_bias[new_M], acc01 = ptr_bias[new_M], acc02 = ptr_bias[new_M],
              acc03 = ptr_bias[new_M];
      for (int ko = 0; ko < K; ko += 16) {
        int8_t frag_B[4][16] = {0};
        for (int ki = 0; ki < 16; ++ki) {
          int16_t tmp;
          memcpy(&tmp, ptr_B + (ko + ki) * N + no, 4);
          frag_B[0][ki] = tmp & 0xff;
          frag_B[1][ki] = (tmp >> 8) & 0xff;
          frag_B[2][ki] = (tmp >> 16) & 0xff;
          frag_B[3][ki] = (tmp >> 24) & 0xff;
        }

        const int8_t* A0 = ptr_A + new_M * K + ko;
        int8_t* B0 = &frag_B[0][0];
        int8_t* B1 = &frag_B[1][0];
        int8_t* B2 = &frag_B[2][0];
        int8_t* B3 = &frag_B[3][0];

        mma_m1n2k16_s8s8s8_acc32_boffset(A0, B0, B1, offset_q15x2, acc00, acc01);
        mma_m1n2k16_s8s8s8_acc32_boffset(A0, B2, B3, offset_q15x2, acc02, acc03);
      }
      int8_t* output00 = ptr_C + new_M * N + no;
      int8_t* output01 = output00 + 1;
      int8_t* output02 = output01 + 1;
      int8_t* output03 = output02 + 1;

      acc00 = (int32_t)((float)(acc00)*scale0) + C_offset;
      acc01 = (int32_t)((float)(acc01)*scale0) + C_offset;
      acc02 = (int32_t)((float)(acc02)*scale0) + C_offset;
      acc03 = (int32_t)((float)(acc03)*scale0) + C_offset;

      acc00 = MIN(clip_max, MAX(clip_min, acc00));
      acc01 = MIN(clip_max, MAX(clip_min, acc01));
      acc02 = MIN(clip_max, MAX(clip_min, acc02));
      acc03 = MIN(clip_max, MAX(clip_min, acc03));

      *(output00) = (int8_t)acc00;
      *(output01) = (int8_t)acc01;
      *(output02) = (int8_t)acc02;
      *(output03) = (int8_t)acc03;
    }
  }
}

}  // namespace mculib

#endif  // MCULIB_MATMUL_MATMUL_MMA_M2N4K16_H