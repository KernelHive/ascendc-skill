
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ReluCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
  TILING_DATA_FIELD_DEF(uint32_t, blockDim);
  TILING_DATA_FIELD_DEF(uint32_t, fullTilesPerBlock);
  TILING_DATA_FIELD_DEF(uint32_t, hasTail);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ReluCustom, ReluCustomTilingData)
}  // namespace optiling
