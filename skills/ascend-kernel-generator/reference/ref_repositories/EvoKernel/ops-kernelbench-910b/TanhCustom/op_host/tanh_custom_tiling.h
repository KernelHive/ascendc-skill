
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TanhCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, blockDim);
  TILING_DATA_FIELD_DEF(uint32_t, tileSize);
  TILING_DATA_FIELD_DEF(uint32_t, tmpSizeBytes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TanhCustom, TanhCustomTilingData)
}  // namespace optiling
