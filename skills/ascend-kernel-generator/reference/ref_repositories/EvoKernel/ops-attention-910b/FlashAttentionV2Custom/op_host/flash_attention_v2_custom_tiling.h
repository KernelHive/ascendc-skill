
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FlashAttentionV2CustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, B);
    TILING_DATA_FIELD_DEF(uint32_t, H);
    TILING_DATA_FIELD_DEF(uint32_t, S);
    TILING_DATA_FIELD_DEF(uint32_t, D);
    TILING_DATA_FIELD_DEF(uint32_t, totalBh);
    TILING_DATA_FIELD_DEF(uint32_t, coreNum);
    TILING_DATA_FIELD_DEF(uint32_t, Ti);
    TILING_DATA_FIELD_DEF(uint32_t, Tj);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FlashAttentionV2Custom, FlashAttentionV2CustomTilingData)
} // namespace optiling
