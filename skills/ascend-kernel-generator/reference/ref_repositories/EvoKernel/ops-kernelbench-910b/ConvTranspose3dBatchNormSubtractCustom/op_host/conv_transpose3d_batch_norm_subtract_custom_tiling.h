
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ConvTranspose3dBatchNormSubtractCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, Cin);
  TILING_DATA_FIELD_DEF(uint32_t, Din);
  TILING_DATA_FIELD_DEF(uint32_t, Hin);
  TILING_DATA_FIELD_DEF(uint32_t, Win);

  TILING_DATA_FIELD_DEF(uint32_t, Cout);
  TILING_DATA_FIELD_DEF(uint32_t, K);

  TILING_DATA_FIELD_DEF(uint32_t, Stride);
  TILING_DATA_FIELD_DEF(uint32_t, Pad);
  TILING_DATA_FIELD_DEF(uint32_t, Dil);
  TILING_DATA_FIELD_DEF(uint32_t, OutPad);

  TILING_DATA_FIELD_DEF(uint32_t, Dout);
  TILING_DATA_FIELD_DEF(uint32_t, Hout);
  TILING_DATA_FIELD_DEF(uint32_t, Wout);

  // For BN and spatial-mean subtraction
  TILING_DATA_FIELD_DEF(uint32_t, DHW);   // Dout*Hout*Wout
  TILING_DATA_FIELD_DEF(uint32_t, NHW);   // N*DHW
  TILING_DATA_FIELD_DEF(float, invDHW);
  TILING_DATA_FIELD_DEF(float, invNHW);
  TILING_DATA_FIELD_DEF(float, eps);

  // runtime
  TILING_DATA_FIELD_DEF(uint32_t, N);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvTranspose3dBatchNormSubtractCustom,
                           ConvTranspose3dBatchNormSubtractCustomTilingData)

}  // namespace optiling
