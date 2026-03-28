
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatrixScalarMultiplicationCustomTilingData)
TILING_DATA_FIELD_DEF(uint64_t, totalLength);
TILING_DATA_FIELD_DEF(uint32_t, tileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatrixScalarMultiplicationCustom, MatrixScalarMultiplicationCustomTilingData)
} // namespace optiling
