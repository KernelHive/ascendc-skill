
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvTransposed1dDilatedCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalY);    // N*Cout*Lout
    TILING_DATA_FIELD_DEF(uint32_t, lout);      // Lout
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);  // launch blocks
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvTransposed1dDilatedCustom,
                           ConvTransposed1dDilatedCustomTilingData)
} // namespace optiling
