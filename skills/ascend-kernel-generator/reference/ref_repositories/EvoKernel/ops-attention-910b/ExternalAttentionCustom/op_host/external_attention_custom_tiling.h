
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ExternalAttentionCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, bs);
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, s);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ExternalAttentionCustom, ExternalAttentionCustomTilingData)
} // namespace optiling
