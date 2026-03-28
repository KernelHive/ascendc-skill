
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvTransposed2dAsymmetricInputSquareKernelCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, rows);      // rows = N*COUT*HOUT
    TILING_DATA_FIELD_DEF(uint32_t, wout);      // WOUT
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);  // launch blocks
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvTransposed2dAsymmetricInputSquareKernelCustom,
                           ConvTransposed2dAsymmetricInputSquareKernelCustomTilingData)
} // namespace optiling
