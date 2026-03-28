
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ECAAttentionCustomTilingData)
    // x: [B,C,H,W], weight: [1,1,K], y: [B,C,H,W]
    TILING_DATA_FIELD_DEF(uint32_t, B);
    TILING_DATA_FIELD_DEF(uint32_t, C);
    TILING_DATA_FIELD_DEF(uint32_t, H);
    TILING_DATA_FIELD_DEF(uint32_t, W);
    TILING_DATA_FIELD_DEF(uint32_t, HW);
    TILING_DATA_FIELD_DEF(uint32_t, K);
    TILING_DATA_FIELD_DEF(uint32_t, pad);
    TILING_DATA_FIELD_DEF(float, invHW);

    // channel tiling used in kernel (pooled is full C; conv/gate is tiled)
    TILING_DATA_FIELD_DEF(uint32_t, cTile);

    // scratch bytes for vector sigmoid primitive (passed as UB uint8)
    TILING_DATA_FIELD_DEF(uint32_t, sigTmpBytes);

    // batch split
    TILING_DATA_FIELD_DEF(uint32_t, totalB);
    TILING_DATA_FIELD_DEF(uint32_t, bPerBlock);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ECAAttentionCustom, ECAAttentionCustomTilingData)
} // namespace optiling
