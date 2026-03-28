
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LinearAttentionCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, b);
    TILING_DATA_FIELD_DEF(uint32_t, h);
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, d);
    TILING_DATA_FIELD_DEF(float, eps);
    TILING_DATA_FIELD_DEF(uint32_t, total_bh);
    TILING_DATA_FIELD_DEF(uint32_t, row_block);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LinearAttentionCustom, LinearAttentionCustomTilingData)
} // namespace optiling
