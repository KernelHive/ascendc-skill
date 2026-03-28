
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvTransposed2dAsymmetricInputAsymmetricKernelStridedGroupedPaddedDilatedCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalX);     // N*Cin*Hin*Win (informational)
    TILING_DATA_FIELD_DEF(uint32_t, totalW);     // Cin*(Cout/groups)*Kh*Kw
    TILING_DATA_FIELD_DEF(uint32_t, totalY);     // N*Cout*Hout*Wout
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);   // launch blocks
    TILING_DATA_FIELD_DEF(uint32_t, tileElems);  // linear y elements per block iteration
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    ConvTransposed2dAsymmetricInputAsymmetricKernelStridedGroupedPaddedDilatedCustom,
    ConvTransposed2dAsymmetricInputAsymmetricKernelStridedGroupedPaddedDilatedCustomTilingData)
} // namespace optiling
