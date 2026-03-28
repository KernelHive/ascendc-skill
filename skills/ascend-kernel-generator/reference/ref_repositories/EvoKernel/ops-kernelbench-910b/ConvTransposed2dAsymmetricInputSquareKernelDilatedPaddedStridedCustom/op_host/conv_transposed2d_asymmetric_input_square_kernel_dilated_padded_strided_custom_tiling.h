
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvTransposed2dAsymmetricInputSquareKernelDilatedPaddedStridedCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalRows); // N*COUT*HOUT
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    ConvTransposed2dAsymmetricInputSquareKernelDilatedPaddedStridedCustom,
    ConvTransposed2dAsymmetricInputSquareKernelDilatedPaddedStridedCustomTilingData)
} // namespace optiling
