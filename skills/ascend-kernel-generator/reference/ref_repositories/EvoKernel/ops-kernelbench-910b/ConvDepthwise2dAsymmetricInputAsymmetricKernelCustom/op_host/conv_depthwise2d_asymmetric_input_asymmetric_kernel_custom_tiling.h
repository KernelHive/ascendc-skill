
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvDepthwise2dAsymmetricInputAsymmetricKernelCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalX); // N*C*H*W
    TILING_DATA_FIELD_DEF(uint32_t, totalW); // C*1*Kh*Kw
    TILING_DATA_FIELD_DEF(uint32_t, totalY); // N*C*Ho*Wo
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvDepthwise2dAsymmetricInputAsymmetricKernelCustom,
                           ConvDepthwise2dAsymmetricInputAsymmetricKernelCustomTilingData)
} // namespace optiling
