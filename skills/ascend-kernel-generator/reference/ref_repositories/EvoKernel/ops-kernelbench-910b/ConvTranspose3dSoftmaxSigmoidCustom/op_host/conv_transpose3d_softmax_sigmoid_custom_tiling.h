
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvTranspose3dSoftmaxSigmoidCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalRows); // N*Dout*Hout*Wout
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvTranspose3dSoftmaxSigmoidCustom,
                           ConvTranspose3dSoftmaxSigmoidCustomTilingData)
} // namespace optiling
