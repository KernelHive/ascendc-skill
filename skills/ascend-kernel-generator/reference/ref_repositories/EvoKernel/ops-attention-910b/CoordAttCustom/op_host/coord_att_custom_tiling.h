
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CoordAttCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalElems);
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);
    TILING_DATA_FIELD_DEF(uint32_t, coreElemsAligned);
    TILING_DATA_FIELD_DEF(uint32_t, tileElems);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CoordAttCustom, CoordAttCustomTilingData)
} // namespace optiling
