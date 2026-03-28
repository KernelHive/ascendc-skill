
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvStandard3dAsymmetricInputAsymmetricKernelCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);
    TILING_DATA_FIELD_DEF(uint32_t, totalRows); // rows = N*COUT*DOUT*HOUT (each row sweeps WOUT)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvStandard3dAsymmetricInputAsymmetricKernelCustom,
                           ConvStandard3dAsymmetricInputAsymmetricKernelCustomTilingData)
} // namespace optiling
