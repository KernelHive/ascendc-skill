
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GeluCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
  TILING_DATA_FIELD_DEF(uint32_t, tanhTmpBytes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GeluCustom, GeluCustomTilingData)
}  // namespace optiling
