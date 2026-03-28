
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvTransposed1dAsymmetricInputSquareKernelPaddedStridedDilatedCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, n);     // 16
    TILING_DATA_FIELD_DEF(uint32_t, cin);   // 32
    TILING_DATA_FIELD_DEF(uint32_t, cout);  // 64
    TILING_DATA_FIELD_DEF(uint32_t, lin);   // 131072
    TILING_DATA_FIELD_DEF(uint32_t, lout);  // 262145
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvTransposed1dAsymmetricInputSquareKernelPaddedStridedDilatedCustom,
                           ConvTransposed1dAsymmetricInputSquareKernelPaddedStridedDilatedCustomTilingData)
} // namespace optiling
