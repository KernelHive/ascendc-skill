
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LeakyReluCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
  TILING_DATA_FIELD_DEF(uint32_t, blockDim);
  TILING_DATA_FIELD_DEF(float, negativeSlope);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LeakyReluCustom, LeakyReluCustomTilingData)
}  // namespace optiling
