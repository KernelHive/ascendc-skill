
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatmulDropoutSoftmaxCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, B);
  TILING_DATA_FIELD_DEF(uint32_t, K);
  TILING_DATA_FIELD_DEF(uint32_t, N);
  TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);
  TILING_DATA_FIELD_DEF(uint32_t, tileN);
  TILING_DATA_FIELD_DEF(uint32_t, kChunkK);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatmulDropoutSoftmaxCustom, MatmulDropoutSoftmaxCustomTilingData)
}  // namespace optiling
