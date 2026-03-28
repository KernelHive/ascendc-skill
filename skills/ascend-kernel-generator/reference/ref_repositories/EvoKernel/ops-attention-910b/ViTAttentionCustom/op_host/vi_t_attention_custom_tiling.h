
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ViTAttentionCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, bs);
    TILING_DATA_FIELD_DEF(uint32_t, heads);
    TILING_DATA_FIELD_DEF(uint32_t, nq);
    TILING_DATA_FIELD_DEF(uint32_t, d);
    TILING_DATA_FIELD_DEF(uint32_t, c);

    // projection tiling
    TILING_DATA_FIELD_DEF(uint32_t, ocTile);
    TILING_DATA_FIELD_DEF(uint32_t, icTile);

    // attention staging
    TILING_DATA_FIELD_DEF(uint32_t, kvStage);        // 1 => stage K/V in UB per (b,h) per token
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ViTAttentionCustom, ViTAttentionCustomTilingData)
} // namespace optiling
