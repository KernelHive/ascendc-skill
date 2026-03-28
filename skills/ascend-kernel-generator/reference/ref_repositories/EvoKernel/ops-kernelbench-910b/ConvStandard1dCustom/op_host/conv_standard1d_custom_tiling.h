
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvStandard1dCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalX);
    TILING_DATA_FIELD_DEF(uint32_t, totalW);
    TILING_DATA_FIELD_DEF(uint32_t, totalY);
    TILING_DATA_FIELD_DEF(uint32_t, blockTiles); // number of y elements per block (tiling hint)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvStandard1dCustom, ConvStandard1dCustomTilingData)
} // namespace optiling
