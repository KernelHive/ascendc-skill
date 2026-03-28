
#include "register/tilingdata_base.h"
#include <cstdint>

namespace optiling {
BEGIN_TILING_DATA_DEF(StreamMixCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, B);
TILING_DATA_FIELD_DEF(uint32_t, T);
TILING_DATA_FIELD_DEF(uint32_t, N);
TILING_DATA_FIELD_DEF(uint32_t, C);
TILING_DATA_FIELD_DEF(uint32_t, BT);     // B*T
TILING_DATA_FIELD_DEF(uint32_t, cTile);  // tile along C for N==4 fastpath (per stream)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(StreamMixCustom, StreamMixCustomTilingData)
} // namespace optiling
