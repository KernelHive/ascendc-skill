
#include "register/tilingdata_base.h"
#include <cstdint>

namespace optiling {
BEGIN_TILING_DATA_DEF(StreamWeightedSumCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, B);
TILING_DATA_FIELD_DEF(uint32_t, T);
TILING_DATA_FIELD_DEF(uint32_t, N);
TILING_DATA_FIELD_DEF(uint32_t, C);
TILING_DATA_FIELD_DEF(uint32_t, BT);          // B*T
TILING_DATA_FIELD_DEF(uint32_t, cTile);       // tile along C, multiple of 8
TILING_DATA_FIELD_DEF(uint32_t, tilesPerRow); // C / cTile (hot-path assumes divisible)
TILING_DATA_FIELD_DEF(uint32_t, numTiles);    // BT * tilesPerRow
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(StreamWeightedSumCustom, StreamWeightedSumCustomTilingData)
} // namespace optiling
