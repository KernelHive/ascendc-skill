
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(Conv2dGroupNormScaleMaxPoolClampCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, Cin);
  TILING_DATA_FIELD_DEF(uint32_t, Hin);
  TILING_DATA_FIELD_DEF(uint32_t, Win);

  TILING_DATA_FIELD_DEF(uint32_t, Cout);
  TILING_DATA_FIELD_DEF(uint32_t, K);

  TILING_DATA_FIELD_DEF(uint32_t, Hc);
  TILING_DATA_FIELD_DEF(uint32_t, Wc);

  TILING_DATA_FIELD_DEF(uint32_t, Ho);
  TILING_DATA_FIELD_DEF(uint32_t, Wo);

  TILING_DATA_FIELD_DEF(uint32_t, G);
  TILING_DATA_FIELD_DEF(uint32_t, CperG);

  TILING_DATA_FIELD_DEF(uint32_t, poolK);
  TILING_DATA_FIELD_DEF(uint32_t, poolS);

  TILING_DATA_FIELD_DEF(uint32_t, elemsPerG);
  TILING_DATA_FIELD_DEF(float, invElemsPerG);
  TILING_DATA_FIELD_DEF(float, eps);

  TILING_DATA_FIELD_DEF(float, clampMin);
  TILING_DATA_FIELD_DEF(float, clampMax);

  // runtime
  TILING_DATA_FIELD_DEF(uint32_t, N);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv2dGroupNormScaleMaxPoolClampCustom,
                           Conv2dGroupNormScaleMaxPoolClampCustomTilingData)

}  // namespace optiling
