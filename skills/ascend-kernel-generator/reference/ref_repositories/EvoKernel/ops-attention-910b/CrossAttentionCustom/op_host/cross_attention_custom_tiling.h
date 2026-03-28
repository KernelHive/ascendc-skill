
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CrossAttentionCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, b);
    TILING_DATA_FIELD_DEF(uint32_t, h);
    TILING_DATA_FIELD_DEF(uint32_t, sq);
    TILING_DATA_FIELD_DEF(uint32_t, sk);
    TILING_DATA_FIELD_DEF(uint32_t, d);
    TILING_DATA_FIELD_DEF(float, scale);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CrossAttentionCustom, CrossAttentionCustomTilingData)
} // namespace optiling
