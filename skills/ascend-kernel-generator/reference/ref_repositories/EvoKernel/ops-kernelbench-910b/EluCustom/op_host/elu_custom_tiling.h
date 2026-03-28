
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(EluCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, blockDim);
  TILING_DATA_FIELD_DEF(uint32_t, tileElems);
  TILING_DATA_FIELD_DEF(float, alpha);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(EluCustom, EluCustomTilingData)
} // namespace optiling
