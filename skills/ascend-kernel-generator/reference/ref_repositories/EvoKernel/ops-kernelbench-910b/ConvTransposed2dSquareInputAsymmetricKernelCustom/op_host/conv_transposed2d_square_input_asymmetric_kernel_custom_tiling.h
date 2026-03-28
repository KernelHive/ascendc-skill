
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvTransposed2dSquareInputAsymmetricKernelCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalRows); // N*COUT*HOUT
    TILING_DATA_FIELD_DEF(uint32_t, wout);      // WOUT
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);  // launch blocks for deterministic partitioning
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvTransposed2dSquareInputAsymmetricKernelCustom,
                           ConvTransposed2dSquareInputAsymmetricKernelCustomTilingData)
} // namespace optiling
