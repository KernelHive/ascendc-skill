
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatmulGeluSoftmaxCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, B);
  TILING_DATA_FIELD_DEF(uint32_t, K);
  TILING_DATA_FIELD_DEF(uint32_t, N);
  TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);
  TILING_DATA_FIELD_DEF(uint32_t, tileN);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatmulGeluSoftmaxCustom, MatmulGeluSoftmaxCustomTilingData)
}  // namespace optiling
