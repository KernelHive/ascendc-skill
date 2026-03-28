
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GatedChannelTransformCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, N);
    TILING_DATA_FIELD_DEF(uint32_t, C);
    TILING_DATA_FIELD_DEF(uint32_t, H);
    TILING_DATA_FIELD_DEF(uint32_t, W);
    TILING_DATA_FIELD_DEF(uint32_t, HW);
    TILING_DATA_FIELD_DEF(uint32_t, CHW);
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF(uint32_t, CAlign);
    TILING_DATA_FIELD_DEF(float, invC);
    TILING_DATA_FIELD_DEF(uint32_t, tanhTmpBytes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GatedChannelTransformCustom, GatedChannelTransformCustomTilingData)
} // namespace optiling
