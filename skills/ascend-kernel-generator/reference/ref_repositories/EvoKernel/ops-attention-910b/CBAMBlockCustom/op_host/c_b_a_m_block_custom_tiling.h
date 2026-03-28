
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CBAMBlockCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, B);
    TILING_DATA_FIELD_DEF(uint32_t, C);
    TILING_DATA_FIELD_DEF(uint32_t, H);
    TILING_DATA_FIELD_DEF(uint32_t, W);
    TILING_DATA_FIELD_DEF(uint32_t, HW);
    TILING_DATA_FIELD_DEF(uint32_t, strideB);  // C*HW
    TILING_DATA_FIELD_DEF(uint32_t, cTile);    // channel tile for cache friendliness
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CBAMBlockCustom, CBAMBlockCustomTilingData)
} // namespace optiling
