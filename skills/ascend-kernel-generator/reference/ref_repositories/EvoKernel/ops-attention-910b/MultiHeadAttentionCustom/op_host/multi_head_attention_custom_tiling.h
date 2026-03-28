
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MultiHeadAttentionCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, b);
    TILING_DATA_FIELD_DEF(uint32_t, h);
    TILING_DATA_FIELD_DEF(uint32_t, s);
    TILING_DATA_FIELD_DEF(uint32_t, d);
    TILING_DATA_FIELD_DEF(float, scale);   // 1/sqrt(d)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MultiHeadAttentionCustom, MultiHeadAttentionCustomTilingData)
} // namespace optiling
