#ifndef MCULIB_MMA_H
#define MCULIB_MMA_H

#include <mem_access.h>
#include <stdint.h>

#include <cstdlib>

namespace mculib {

#define MMA_M1N2K4_ROUND_BOFFSET     \
  LOADs8x4(b01, ptr_B0);             \
  ptr_B0 += 4;                       \
  b02 = __SXTB16(__ROR(b01, 8));     \
  b01 = __SXTB16(b01);               \
                                     \
  b02 = __SADD16(b02, offset_q15x2); \
  b01 = __SADD16(b01, offset_q15x2); \
                                     \
  LOADs8x4(a01, ptr_A0);             \
  ptr_A0 += 4;                       \
  a02 = __SXTB16(__ROR(a01, 8));     \
  a01 = __SXTB16(a01);               \
                                     \
  acc00 = __SMLAD(a01, b01, acc00);  \
  acc00 = __SMLAD(a02, b02, acc00);  \
                                     \
  LOADs8x4(b11, ptr_B1);             \
  ptr_B1 += 4;                       \
  b12 = __SXTB16(__ROR(b11, 8));     \
  b11 = __SXTB16(b11);               \
                                     \
  b12 = __SADD16(b12, offset_q15x2); \
  b11 = __SADD16(b11, offset_q15x2); \
                                     \
  acc01 = __SMLAD(a01, b11, acc01);  \
  acc01 = __SMLAD(a02, b12, acc01);

#define MMA_M2N2K4_ROUND_BOFFSET     \
  LOADs8x4(b01, ptr_B0);             \
  ptr_B0 += 4;                       \
  b02 = __SXTB16(__ROR(b01, 8));     \
  b01 = __SXTB16(b01);               \
                                     \
  b02 = __SADD16(b02, offset_q15x2); \
  b01 = __SADD16(b01, offset_q15x2); \
                                     \
  acc11 = __SMLAD(a11, b11, acc11);  \
  acc11 = __SMLAD(a12, b12, acc11);  \
                                     \
  LOADs8x4(a01, ptr_A0);             \
  ptr_A0 += 4;                       \
  a02 = __SXTB16(__ROR(a01, 8));     \
  a01 = __SXTB16(a01);               \
                                     \
  acc00 = __SMLAD(a01, b01, acc00);  \
  acc00 = __SMLAD(a02, b02, acc00);  \
                                     \
  LOADs8x4(b11, ptr_B1);             \
  ptr_B1 += 4;                       \
  b12 = __SXTB16(__ROR(b11, 8));     \
  b11 = __SXTB16(b11);               \
                                     \
  b12 = __SADD16(b12, offset_q15x2); \
  b11 = __SADD16(b11, offset_q15x2); \
                                     \
  acc01 = __SMLAD(a01, b11, acc01);  \
  acc01 = __SMLAD(a02, b12, acc01);  \
                                     \
  LOADs8x4(a11, ptr_A1);             \
  ptr_A1 += 4;                       \
  a12 = __SXTB16(__ROR(a11, 8));     \
  a11 = __SXTB16(a11);               \
                                     \
  acc10 = __SMLAD(a11, b01, acc10);  \
  acc10 = __SMLAD(a12, b02, acc10);

#define MMA_M2N2K4_ROUND_AOFFSET     \
  LOADs8x4(a01, ptr_A0);             \
  ptr_A0 += 4;                       \
  a02 = __SXTB16(__ROR(a01, 8));     \
  a01 = __SXTB16(a01);               \
                                     \
  a02 = __SADD16(a02, offset_q15x2); \
  a01 = __SADD16(a01, offset_q15x2); \
                                     \
  acc11 = __SMLAD(a11, b11, acc11);  \
  acc11 = __SMLAD(a12, b12, acc11);  \
                                     \
  LOADs8x4(b01, ptr_B0);             \
  ptr_B0 += 4;                       \
  b02 = __SXTB16(__ROR(b01, 8));     \
  b01 = __SXTB16(b01);               \
                                     \
  acc00 = __SMLAD(a01, b01, acc00);  \
  acc00 = __SMLAD(a02, b02, acc00);  \
                                     \
  LOADs8x4(a11, ptr_A1);             \
  ptr_A1 += 4;                       \
  a12 = __SXTB16(__ROR(a11, 8));     \
  a11 = __SXTB16(a11);               \
                                     \
  a12 = __SADD16(a12, offset_q15x2); \
  a11 = __SADD16(a11, offset_q15x2); \
                                     \
  acc01 = __SMLAD(a01, b11, acc01);  \
  acc01 = __SMLAD(a02, b12, acc01);  \
                                     \
  LOADs8x4(b11, ptr_B1);             \
  ptr_B1 += 4;                       \
  a12 = __SXTB16(__ROR(b11, 8));     \
  a11 = __SXTB16(b11);               \
                                     \
  acc10 = __SMLAD(a11, b01, acc10);  \
  acc10 = __SMLAD(a12, b02, acc10);

inline void mma_m1n2k8_s8s8s8_acc32_boffset(int8_t* ptr_A0, int8_t* ptr_B0, int8_t* ptr_B1,
                                            int32_t offset_q15x2, int32_t& acc00, int32_t& acc01) {
  int32_t a01, a02;
  int32_t b01, b02, b11, b12;

  MMA_M1N2K4_ROUND_BOFFSET
}

/**
 * A: [2, 8]
 * B: [2, 8]
 * Output: [2, 2]
 *
 * A is weight passed by two pointers (A0,A1)
 * B is activation passed by two pointers (B0,B1)
 * Output is passed by reference (acc00,01,10,11)
 */
inline void mma_m2n2k8_s8s8s8_acc32_boffset(int8_t* ptr_A0, int8_t* ptr_B0, int8_t* ptr_A1,
                                            int8_t* ptr_B1, int32_t offset_q15x2, int32_t& acc00,
                                            int32_t& acc01, int32_t& acc10, int32_t& acc11

) {
  int32_t a01, a02, a11, a12;
  int32_t b01, b02, b11, b12;

  LOADs8x4(b01, ptr_B0);
  ptr_B0 += 4;
  b02 = __SXTB16(__ROR(b01, 8));
  b01 = __SXTB16(b01);

  b02 = __SADD16(b02, offset_q15x2);
  b01 = __SADD16(b01, offset_q15x2);

  LOADs8x4(a01, ptr_A0);
  ptr_A0 += 4;
  a02 = __SXTB16(__ROR(a01, 8));
  a01 = __SXTB16(a01);

  acc00 = __SMLAD(a01, b01, acc00);
  acc00 = __SMLAD(a02, b02, acc00);

  LOADs8x4(b11, ptr_B1);
  ptr_B1 += 4;
  b12 = __SXTB16(__ROR(b11, 8));
  b11 = __SXTB16(b11);

  b12 = __SADD16(b12, offset_q15x2);
  b11 = __SADD16(b11, offset_q15x2);

  acc01 = __SMLAD(a01, b11, acc01);
  acc01 = __SMLAD(a02, b12, acc01);

  LOADs8x4(a11, ptr_A1);
  ptr_A1 += 4;
  a12 = __SXTB16(__ROR(a11, 8));
  a11 = __SXTB16(a11);

  acc10 = __SMLAD(a11, b01, acc10);
  acc10 = __SMLAD(a12, b02, acc10);

  // prefetch
  MMA_M2N2K4_ROUND_BOFFSET

  // no prefetch
  acc11 = __SMLAD(a11, b11, acc11);
  acc11 = __SMLAD(a12, b12, acc11);
}

inline void mma_m1n2k16_s8s8s8_acc32_boffset(const int8_t* ptr_A0, int8_t* ptr_B0, int8_t* ptr_B1,
                                             int32_t offset_q15x2, int32_t& acc00, int32_t& acc01) {
  int32_t a01, a02;
  int32_t b01, b02, b11, b12;

  MMA_M1N2K4_ROUND_BOFFSET
  MMA_M1N2K4_ROUND_BOFFSET
}

inline void mma_m2n2k16_s8s8s8_acc32_boffset(const int8_t* ptr_A0, int8_t* ptr_B0,
                                             const int8_t* ptr_A1, int8_t* ptr_B1,
                                             int32_t offset_q15x2, int32_t& acc00, int32_t& acc01,
                                             int32_t& acc10, int32_t& acc11) {
  int32_t a01, a02, a11, a12;
  int32_t b01, b02, b11, b12;

  LOADs8x4(b01, ptr_B0);
  ptr_B0 += 4;
  b02 = __SXTB16(__ROR(b01, 8));
  b01 = __SXTB16(b01);

  b02 = __SADD16(b02, offset_q15x2);
  b01 = __SADD16(b01, offset_q15x2);

  LOADs8x4(a01, ptr_A0);
  ptr_A0 += 4;
  a02 = __SXTB16(__ROR(a01, 8));
  a01 = __SXTB16(a01);

  acc00 = __SMLAD(a01, b01, acc00);
  acc00 = __SMLAD(a02, b02, acc00);

  LOADs8x4(b11, ptr_B1);
  ptr_B1 += 4;
  b12 = __SXTB16(__ROR(b11, 8));
  b11 = __SXTB16(b11);

  b12 = __SADD16(b12, offset_q15x2);
  b11 = __SADD16(b11, offset_q15x2);

  acc01 = __SMLAD(a01, b11, acc01);
  acc01 = __SMLAD(a02, b12, acc01);

  LOADs8x4(a11, ptr_A1);
  ptr_A1 += 4;
  a12 = __SXTB16(__ROR(a11, 8));
  a11 = __SXTB16(a11);

  acc10 = __SMLAD(a11, b01, acc10);
  acc10 = __SMLAD(a12, b02, acc10);

  // prefetch
  MMA_M2N2K4_ROUND_BOFFSET
  // prefetch
  MMA_M2N2K4_ROUND_BOFFSET
  // prefetch
  MMA_M2N2K4_ROUND_BOFFSET

  // no prefetch
  acc11 = __SMLAD(a11, b11, acc11);
  acc11 = __SMLAD(a12, b12, acc11);
}

inline void mma_m2n2k16_s8s8s8_acc32_aoffset(int8_t* ptr_A0, const int8_t* ptr_B0, int8_t* ptr_A1,
                                             const int8_t* ptr_B1, int32_t offset_q15x2,
                                             int32_t& acc00, int32_t& acc01, int32_t& acc10,
                                             int32_t& acc11) {
  int32_t a01, a02, a11, a12;
  int32_t b01, b02, b11, b12;

  LOADs8x4(a01, ptr_A0);
  ptr_A0 += 4;
  a02 = __SXTB16(__ROR(a01, 8));
  a01 = __SXTB16(a01);

  a02 = __SADD16(a02, offset_q15x2);
  a01 = __SADD16(a01, offset_q15x2);

  LOADs8x4(b01, ptr_B0);
  ptr_B0 += 4;
  b02 = __SXTB16(__ROR(b01, 8));
  b01 = __SXTB16(b01);

  acc00 = __SMLAD(a01, b01, acc00);
  acc00 = __SMLAD(a02, b02, acc00);

  LOADs8x4(a11, ptr_A1);
  ptr_A1 += 4;
  a12 = __SXTB16(__ROR(a11, 8));
  a11 = __SXTB16(a11);

  a12 = __SADD16(a12, offset_q15x2);
  a11 = __SADD16(a11, offset_q15x2);

  acc01 = __SMLAD(a01, b11, acc01);
  acc01 = __SMLAD(a02, b12, acc01);

  LOADs8x4(b11, ptr_B1);
  ptr_B1 += 4;
  b12 = __SXTB16(__ROR(b11, 8));
  b11 = __SXTB16(b11);

  acc10 = __SMLAD(a11, b01, acc10);
  acc10 = __SMLAD(a12, b02, acc10);

  // prefetch
  MMA_M2N2K4_ROUND_AOFFSET
  // prefetch
  MMA_M2N2K4_ROUND_AOFFSET
  // prefetch
  MMA_M2N2K4_ROUND_AOFFSET

  // no prefetch
  acc11 = __SMLAD(a11, b11, acc11);
  acc11 = __SMLAD(a12, b12, acc11);
}

inline void mma_m2n2k144_s8s8s8_acc32_boffset(const int8_t* ptr_A0, int8_t* ptr_B0,
                                              const int8_t* ptr_A1, int8_t* ptr_B1,
                                              int32_t offset_q15x2, int32_t& acc00, int32_t& acc01,
                                              int32_t& acc10, int32_t& acc11

) {
  int32_t a01, a02, a11, a12;
  int32_t b01, b02, b11, b12;

  for (int i = 0; i < 9; ++i) {
    LOADs8x4(b01, ptr_B0);
    ptr_B0 += 4;
    b02 = __SXTB16(__ROR(b01, 8));
    b01 = __SXTB16(b01);

    b02 = __SADD16(b02, offset_q15x2);
    b01 = __SADD16(b01, offset_q15x2);

    LOADs8x4(a01, ptr_A0);
    ptr_A0 += 4;
    a02 = __SXTB16(__ROR(a01, 8));
    a01 = __SXTB16(a01);

    acc00 = __SMLAD(a01, b01, acc00);
    acc00 = __SMLAD(a02, b02, acc00);

    LOADs8x4(b11, ptr_B1);
    ptr_B1 += 4;
    b12 = __SXTB16(__ROR(b11, 8));
    b11 = __SXTB16(b11);

    b12 = __SADD16(b12, offset_q15x2);
    b11 = __SADD16(b11, offset_q15x2);

    acc01 = __SMLAD(a01, b11, acc01);
    acc01 = __SMLAD(a02, b12, acc01);

    LOADs8x4(a11, ptr_A1);
    ptr_A1 += 4;
    a12 = __SXTB16(__ROR(a11, 8));
    a11 = __SXTB16(a11);

    acc10 = __SMLAD(a11, b01, acc10);
    acc10 = __SMLAD(a12, b02, acc10);

    // prefetch
    MMA_M2N2K4_ROUND_BOFFSET
    // prefetch
    MMA_M2N2K4_ROUND_BOFFSET
    // prefetch
    MMA_M2N2K4_ROUND_BOFFSET

    // no prefetch
    acc11 = __SMLAD(a11, b11, acc11);
    acc11 = __SMLAD(a12, b12, acc11);
  }
}

}  // namespace mculib

#endif  // MCULIB_MMA_H