
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LogSoftmaxCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, rows);
  TILING_DATA_FIELD_DEF(uint32_t, cols);
  TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);
  TILING_DATA_FIELD_DEF(uint32_t, tileCols);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LogSoftmaxCustom, LogSoftmaxCustomTilingData)
}  // namespace optiling
