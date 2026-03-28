
#include "register/tilingdata_base.h"
#include <cstdint>

namespace optiling {
BEGIN_TILING_DATA_DEF(MhcUpdateCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, B);
TILING_DATA_FIELD_DEF(uint32_t, T);
TILING_DATA_FIELD_DEF(uint32_t, I);
TILING_DATA_FIELD_DEF(uint32_t, J);
TILING_DATA_FIELD_DEF(uint32_t, C);
TILING_DATA_FIELD_DEF(uint32_t, BT);   // B*T
TILING_DATA_FIELD_DEF(uint32_t, Vc);   // tile along C in floats, multiple of 8 for fp32
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MhcUpdateCustom, MhcUpdateCustomTilingData)
} // namespace optiling
