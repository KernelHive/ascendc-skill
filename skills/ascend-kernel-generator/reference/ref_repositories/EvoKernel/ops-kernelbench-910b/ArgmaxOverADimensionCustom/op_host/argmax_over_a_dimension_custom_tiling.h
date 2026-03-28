
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ArgmaxOverADimensionCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalX);
    TILING_DATA_FIELD_DEF(uint32_t, totalY);
    TILING_DATA_FIELD_DEF(uint32_t, batch);
    TILING_DATA_FIELD_DEF(uint32_t, reduceDim);
    TILING_DATA_FIELD_DEF(uint32_t, innerDim);
    TILING_DATA_FIELD_DEF(uint32_t, tileInner);
    TILING_DATA_FIELD_DEF(uint32_t, tilesPerBatch);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ArgmaxOverADimensionCustom,
                           ArgmaxOverADimensionCustomTilingData)
} // namespace optiling
