
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(L2NormCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, rows);       // folded rows (all dims except axis=1)
  TILING_DATA_FIELD_DEF(uint32_t, cols);       // axis=1 length
  TILING_DATA_FIELD_DEF(uint32_t, tileLength); // UB tile (elements)
  TILING_DATA_FIELD_DEF(float, eps);           // numerical safety
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(L2NormCustom, L2NormCustomTilingData)
}  // namespace optiling
