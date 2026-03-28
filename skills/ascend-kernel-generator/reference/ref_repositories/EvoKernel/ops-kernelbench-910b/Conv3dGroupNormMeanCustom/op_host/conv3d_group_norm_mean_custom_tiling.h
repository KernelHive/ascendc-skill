
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(Conv3dGroupNormMeanCustomTilingData)
  // baked specialization parameters
  TILING_DATA_FIELD_DEF(uint32_t, Cin);
  TILING_DATA_FIELD_DEF(uint32_t, Din);
  TILING_DATA_FIELD_DEF(uint32_t, Hin);
  TILING_DATA_FIELD_DEF(uint32_t, Win);

  TILING_DATA_FIELD_DEF(uint32_t, Cout);
  TILING_DATA_FIELD_DEF(uint32_t, K);

  TILING_DATA_FIELD_DEF(uint32_t, Dout);
  TILING_DATA_FIELD_DEF(uint32_t, Hout);
  TILING_DATA_FIELD_DEF(uint32_t, Wout);

  TILING_DATA_FIELD_DEF(uint32_t, G);
  TILING_DATA_FIELD_DEF(uint32_t, CperG);

  TILING_DATA_FIELD_DEF(uint32_t, DHW);        // Dout*Hout*Wout
  TILING_DATA_FIELD_DEF(uint32_t, elemsPerG);  // CperG*DHW
  TILING_DATA_FIELD_DEF(uint32_t, elemsPerN);  // Cout*DHW

  // kept for ABI compatibility (not performance critical after kernel rewrite)
  TILING_DATA_FIELD_DEF(uint32_t, tileDhw);

  TILING_DATA_FIELD_DEF(float, invElemsPerG);
  TILING_DATA_FIELD_DEF(float, invElemsPerN);
  TILING_DATA_FIELD_DEF(float, eps);

  // runtime
  TILING_DATA_FIELD_DEF(uint32_t, N);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv3dGroupNormMeanCustom, Conv3dGroupNormMeanCustomTilingData)

}  // namespace optiling
