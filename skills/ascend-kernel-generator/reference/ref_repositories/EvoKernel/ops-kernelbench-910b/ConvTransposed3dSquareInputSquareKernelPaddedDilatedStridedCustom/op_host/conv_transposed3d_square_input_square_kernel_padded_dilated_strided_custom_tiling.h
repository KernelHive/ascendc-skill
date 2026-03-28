
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvTransposed3dSquareInputSquareKernelPaddedDilatedStridedCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalX);
    TILING_DATA_FIELD_DEF(uint32_t, totalW);
    TILING_DATA_FIELD_DEF(uint32_t, totalY);
    TILING_DATA_FIELD_DEF(uint32_t, rows);
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvTransposed3dSquareInputSquareKernelPaddedDilatedStridedCustom,
                           ConvTransposed3dSquareInputSquareKernelPaddedDilatedStridedCustomTilingData)
} // namespace optiling
