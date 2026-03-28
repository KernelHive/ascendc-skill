
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MinReductionOverADimensionCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, reduceDim);
TILING_DATA_FIELD_DEF(uint32_t, innerDim);
TILING_DATA_FIELD_DEF(uint32_t, outerCount);
TILING_DATA_FIELD_DEF(uint32_t, colsPerCore);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MinReductionOverADimensionCustom, MinReductionOverADimensionCustomTilingData)
} // namespace optiling
