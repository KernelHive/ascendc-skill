
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvStandard3dSquareInputSquareKernelCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalX);    // N * Cin * D * H * W
    TILING_DATA_FIELD_DEF(uint32_t, totalW);    // Cout * Cin * K * K * K
    TILING_DATA_FIELD_DEF(uint32_t, totalY);    // N * Cout * Do * Ho * Wo
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);  // pass launch blockDim for deterministic partitioning
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvStandard3dSquareInputSquareKernelCustom,
                           ConvStandard3dSquareInputSquareKernelCustomTilingData)
} // namespace optiling
