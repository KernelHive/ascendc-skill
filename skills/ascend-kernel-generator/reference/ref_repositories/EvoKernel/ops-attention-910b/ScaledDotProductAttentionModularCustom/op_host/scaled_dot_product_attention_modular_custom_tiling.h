
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ScaledDotProductAttentionModularCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, b);
    TILING_DATA_FIELD_DEF(uint32_t, h);
    TILING_DATA_FIELD_DEF(uint32_t, nq);
    TILING_DATA_FIELD_DEF(uint32_t, nk);
    TILING_DATA_FIELD_DEF(uint32_t, dk);
    TILING_DATA_FIELD_DEF(uint32_t, dv);
    TILING_DATA_FIELD_DEF(float, scale);        // 1/sqrt(dk)
    TILING_DATA_FIELD_DEF(uint32_t, block_dim); // launch blocks
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScaledDotProductAttentionModularCustom,
                           ScaledDotProductAttentionModularCustomTilingData)
} // namespace optiling
