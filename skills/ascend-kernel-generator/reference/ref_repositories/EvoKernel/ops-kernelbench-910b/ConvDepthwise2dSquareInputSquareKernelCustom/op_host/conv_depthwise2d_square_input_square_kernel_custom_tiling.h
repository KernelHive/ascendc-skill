
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvDepthwise2dSquareInputSquareKernelCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalY);
    TILING_DATA_FIELD_DEF(uint32_t, rows);     // rows = N*C*OH
    TILING_DATA_FIELD_DEF(uint32_t, ow);       // OW
    TILING_DATA_FIELD_DEF(uint32_t, h);        // H
    TILING_DATA_FIELD_DEF(uint32_t, w);        // W
    TILING_DATA_FIELD_DEF(uint32_t, c);        // C
    TILING_DATA_FIELD_DEF(uint32_t, oh);       // OH
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvDepthwise2dSquareInputSquareKernelCustom,
                           ConvDepthwise2dSquareInputSquareKernelCustomTilingData)
} // namespace optiling
