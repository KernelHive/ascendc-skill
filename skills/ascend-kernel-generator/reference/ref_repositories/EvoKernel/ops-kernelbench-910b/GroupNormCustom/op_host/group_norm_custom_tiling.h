
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GroupNormCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, N);
  TILING_DATA_FIELD_DEF(uint32_t, C);
  TILING_DATA_FIELD_DEF(uint32_t, HW);
  TILING_DATA_FIELD_DEF(uint32_t, G);
  TILING_DATA_FIELD_DEF(uint32_t, groupSize);
  TILING_DATA_FIELD_DEF(uint32_t, groupsTotal); // N * G
  TILING_DATA_FIELD_DEF(float, invReduce);      // 1.0f / (groupSize * HW)
  TILING_DATA_FIELD_DEF(float, eps);            // baked
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupNormCustom, GroupNormCustomTilingData)

}  // namespace optiling
