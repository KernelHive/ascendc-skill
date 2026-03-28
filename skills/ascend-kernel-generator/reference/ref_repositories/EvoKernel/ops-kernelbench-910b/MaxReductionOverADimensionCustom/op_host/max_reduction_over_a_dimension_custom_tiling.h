
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MaxReductionOverADimensionCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalX);
    TILING_DATA_FIELD_DEF(uint32_t, totalY);
    TILING_DATA_FIELD_DEF(uint32_t, batch);
    TILING_DATA_FIELD_DEF(uint32_t, reduceDim);
    TILING_DATA_FIELD_DEF(uint32_t, innerDim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MaxReductionOverADimensionCustom,
                           MaxReductionOverADimensionCustomTilingData)
} // namespace optiling
