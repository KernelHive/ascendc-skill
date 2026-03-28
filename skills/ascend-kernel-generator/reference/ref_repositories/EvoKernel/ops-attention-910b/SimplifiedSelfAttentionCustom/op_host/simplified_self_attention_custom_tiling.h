
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SimplifiedSelfAttentionCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, b);
    TILING_DATA_FIELD_DEF(uint32_t, h);
    TILING_DATA_FIELD_DEF(uint32_t, nq);
    TILING_DATA_FIELD_DEF(uint32_t, nk);
    TILING_DATA_FIELD_DEF(uint32_t, d);
    TILING_DATA_FIELD_DEF(float, scale);        // 1/sqrt(d)
    TILING_DATA_FIELD_DEF(uint32_t, block_dim); // launch blocks (capped)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SimplifiedSelfAttentionCustom, SimplifiedSelfAttentionCustomTilingData)
} // namespace optiling
