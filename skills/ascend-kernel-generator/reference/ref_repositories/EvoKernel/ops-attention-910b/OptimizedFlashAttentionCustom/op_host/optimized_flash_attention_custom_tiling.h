
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(OptimizedFlashAttentionCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, b);
    TILING_DATA_FIELD_DEF(uint32_t, h);
    TILING_DATA_FIELD_DEF(uint32_t, sq);
    TILING_DATA_FIELD_DEF(uint32_t, sk);
    TILING_DATA_FIELD_DEF(uint32_t, d);
    TILING_DATA_FIELD_DEF(float, scale);              // 1/sqrt(d)
    TILING_DATA_FIELD_DEF(uint32_t, hasBias);
    TILING_DATA_FIELD_DEF(uint32_t, biasBroadcastB);  // 1 if bias provided as [H,Sq,Sk] and broadcast over B in kernel
    TILING_DATA_FIELD_DEF(uint32_t, tileD);           // UB tile for D dimension
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(OptimizedFlashAttentionCustom, OptimizedFlashAttentionCustomTilingData)
} // namespace optiling
