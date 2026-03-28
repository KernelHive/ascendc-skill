
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(RmsNormCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, b);
  TILING_DATA_FIELD_DEF(uint32_t, c);
  TILING_DATA_FIELD_DEF(uint32_t, inner);
  TILING_DATA_FIELD_DEF(uint32_t, innerTile);
  TILING_DATA_FIELD_DEF(uint32_t, tilesPerB);
  TILING_DATA_FIELD_DEF(float, eps);
  TILING_DATA_FIELD_DEF(float, invC);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RmsNormCustom, RmsNormCustomTilingData)
}  // namespace optiling
