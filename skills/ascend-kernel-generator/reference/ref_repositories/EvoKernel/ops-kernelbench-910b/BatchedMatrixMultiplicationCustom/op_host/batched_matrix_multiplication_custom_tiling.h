
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(BatchedMatrixMultiplicationCustomTilingData)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingData);
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, tilesPerBlockN); // small N-tile grouping factor
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchedMatrixMultiplicationCustom, BatchedMatrixMultiplicationCustomTilingData)
} // namespace optiling
