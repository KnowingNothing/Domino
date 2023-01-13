#ifndef MCULIB_MEM_ACCESS_H
#define MCULIB_MEM_ACCESS_H

#include "mbed.h"
#include <cstdlib>

namespace mculib {

#define LOADs8x4(val, addr) memcpy(&(val), (addr), 4);
#define STOREs8x4(val, addr) (*((int32_t *)(addr)) = val);

} // namespace mculib

#endif // MCULIB_MEM_ACCESS_H