
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Conv3dMinSoftmaxCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalX);
    TILING_DATA_FIELD_DEF(uint32_t, totalW);
    TILING_DATA_FIELD_DEF(uint32_t, totalB);
    TILING_DATA_FIELD_DEF(uint32_t, totalY);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv3dMinSoftmaxCustom,
                           Conv3dMinSoftmaxCustomTilingData)
} // namespace optiling
