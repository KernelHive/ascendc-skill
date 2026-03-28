
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(InstanceNormCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, n);
  TILING_DATA_FIELD_DEF(uint32_t, c);
  TILING_DATA_FIELD_DEF(uint32_t, hw);
  TILING_DATA_FIELD_DEF(uint32_t, planes);

  // Multi-core sharding
  TILING_DATA_FIELD_DEF(uint32_t, blockDim);

  // Tile over HW (elements per tile)
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);

  // Reduction scratch length (aligned, in elements)
  TILING_DATA_FIELD_DEF(uint32_t, reduceTmpLen);

  // Constants
  TILING_DATA_FIELD_DEF(float, invHw);
  TILING_DATA_FIELD_DEF(float, eps);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(InstanceNormCustom, InstanceNormCustomTilingData)
}  // namespace optiling
