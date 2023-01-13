#ifndef MCULIB_PACK_VALUE_H
#define MCULIB_PACK_VALUE_H

namespace mculib {

#define MIN(a, b) (a) > (b) ? (b) : (a)
#define MAX(a, b) (a) > (b) ? (a) : (b)
#define PACKs16x2(v1, v2) __PKHBT((v1), (v2), 16)

} // namespace mculib

#endif // MCULIB_PACK_VALUE_H