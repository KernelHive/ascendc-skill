
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MeanReductionOverADimensionCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batch);
    TILING_DATA_FIELD_DEF(uint32_t, reduceDim);
    TILING_DATA_FIELD_DEF(uint32_t, innerDim);
    TILING_DATA_FIELD_DEF(uint32_t, outerCount); // batch * innerDim
    TILING_DATA_FIELD_DEF(float, invReduce);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MeanReductionOverADimensionCustom,
                           MeanReductionOverADimensionCustomTilingData)
} // namespace optiling
