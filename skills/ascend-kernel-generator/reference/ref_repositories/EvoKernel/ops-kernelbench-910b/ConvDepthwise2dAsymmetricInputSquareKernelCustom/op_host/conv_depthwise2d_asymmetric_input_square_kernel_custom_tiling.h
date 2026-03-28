
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvDepthwise2dAsymmetricInputSquareKernelCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, c);
    TILING_DATA_FIELD_DEF(uint32_t, h);
    TILING_DATA_FIELD_DEF(uint32_t, w);
    TILING_DATA_FIELD_DEF(uint32_t, oh);
    TILING_DATA_FIELD_DEF(uint32_t, ow);
    TILING_DATA_FIELD_DEF(uint32_t, tile_ow);
    TILING_DATA_FIELD_DEF(uint32_t, ow_tiles);
    TILING_DATA_FIELD_DEF(uint32_t, rows);   // rows = N*C*OH
    TILING_DATA_FIELD_DEF(uint32_t, tasks);  // tasks = rows * ow_tiles
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvDepthwise2dAsymmetricInputSquareKernelCustom,
                           ConvDepthwise2dAsymmetricInputSquareKernelCustomTilingData)
} // namespace optiling
