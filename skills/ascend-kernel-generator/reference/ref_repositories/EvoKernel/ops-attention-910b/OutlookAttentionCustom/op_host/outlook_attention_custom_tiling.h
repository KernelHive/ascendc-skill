
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(OutlookAttentionCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, B);
    TILING_DATA_FIELD_DEF(uint32_t, NH);
    TILING_DATA_FIELD_DEF(uint32_t, HD);
    TILING_DATA_FIELD_DEF(uint32_t, C);

    TILING_DATA_FIELD_DEF(uint32_t, H);
    TILING_DATA_FIELD_DEF(uint32_t, W);
    TILING_DATA_FIELD_DEF(uint32_t, HW);

    TILING_DATA_FIELD_DEF(uint32_t, K);
    TILING_DATA_FIELD_DEF(uint32_t, P);
    TILING_DATA_FIELD_DEF(uint32_t, S);
    TILING_DATA_FIELD_DEF(uint32_t, K2);

    TILING_DATA_FIELD_DEF(uint32_t, Ho);
    TILING_DATA_FIELD_DEF(uint32_t, Wo);
    TILING_DATA_FIELD_DEF(uint32_t, HWo);

    // Work partition: each core gets a range of (b, c0) tiles.
    TILING_DATA_FIELD_DEF(uint32_t, cTile);      // channels per tile
    TILING_DATA_FIELD_DEF(uint32_t, tilesPerB);  // ceil(C/cTile)
    TILING_DATA_FIELD_DEF(uint32_t, totalTiles); // B*tilesPerB

    TILING_DATA_FIELD_DEF(uint32_t, coreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(OutlookAttentionCustom, OutlookAttentionCustomTilingData)
} // namespace optiling
