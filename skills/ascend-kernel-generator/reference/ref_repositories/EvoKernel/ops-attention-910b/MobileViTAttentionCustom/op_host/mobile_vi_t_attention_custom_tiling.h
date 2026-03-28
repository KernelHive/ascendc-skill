
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MobileViTAttentionCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalElems);
    TILING_DATA_FIELD_DEF(uint32_t, totalAligned);
    TILING_DATA_FIELD_DEF(uint32_t, tileElems);
    TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MobileViTAttentionCustom, MobileViTAttentionCustomTilingData)
} // namespace optiling
