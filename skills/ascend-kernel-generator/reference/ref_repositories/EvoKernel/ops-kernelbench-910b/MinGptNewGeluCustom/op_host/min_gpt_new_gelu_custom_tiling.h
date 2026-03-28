
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MinGptNewGeluCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
  TILING_DATA_FIELD_DEF(uint32_t, tiles);
  TILING_DATA_FIELD_DEF(uint32_t, tanhTmpBytes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MinGptNewGeluCustom, MinGptNewGeluCustomTilingData)
}  // namespace optiling
