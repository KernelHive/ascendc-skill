
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LayerNormCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, rows);
  TILING_DATA_FIELD_DEF(uint32_t, cols);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
  TILING_DATA_FIELD_DEF(float, eps);
  TILING_DATA_FIELD_DEF(float, invCols);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormCustom, LayerNormCustomTilingData)
}  // namespace optiling
