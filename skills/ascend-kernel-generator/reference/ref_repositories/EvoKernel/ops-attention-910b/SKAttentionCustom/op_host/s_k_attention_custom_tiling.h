
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SKAttentionCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, K);
    TILING_DATA_FIELD_DEF(uint32_t, bs);
    TILING_DATA_FIELD_DEF(uint32_t, C);
    TILING_DATA_FIELD_DEF(uint32_t, H);
    TILING_DATA_FIELD_DEF(uint32_t, W);
    TILING_DATA_FIELD_DEF(uint32_t, hw);           // H*W
    TILING_DATA_FIELD_DEF(uint32_t, totalOut);     // bs*C*H*W
    TILING_DATA_FIELD_DEF(uint32_t, coreNum);
    TILING_DATA_FIELD_DEF(uint32_t, coreStart);    // per-core start (in elements of totalOut)
    TILING_DATA_FIELD_DEF(uint32_t, coreCount);    // per-core count (<= totalOut)
    TILING_DATA_FIELD_DEF(uint32_t, tileNum);
    TILING_DATA_FIELD_DEF(uint32_t, tileElems);    // elements per tile (aligned)
    TILING_DATA_FIELD_DEF(uint32_t, lastTileElems);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SKAttentionCustom, SKAttentionCustomTilingData)
} // namespace optiling
