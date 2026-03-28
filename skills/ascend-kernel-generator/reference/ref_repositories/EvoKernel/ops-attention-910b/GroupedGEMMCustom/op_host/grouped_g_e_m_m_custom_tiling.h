
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GroupedGEMMCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, m);
    TILING_DATA_FIELD_DEF(uint32_t, k);
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, g);
    TILING_DATA_FIELD_DEF(uint32_t, nTile);
    TILING_DATA_FIELD_DEF(uint32_t, kTile);
    TILING_DATA_FIELD_DEF(uint32_t, tilesPerRow);
    TILING_DATA_FIELD_DEF(uint32_t, totalTiles);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupedGEMMCustom, GroupedGEMMCustomTilingData)
} // namespace optiling
