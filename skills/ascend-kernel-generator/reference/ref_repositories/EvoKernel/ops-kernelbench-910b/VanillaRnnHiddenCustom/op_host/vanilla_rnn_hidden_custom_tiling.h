
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(VanillaRnnHiddenCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalX);
    TILING_DATA_FIELD_DEF(uint32_t, totalH0);
    TILING_DATA_FIELD_DEF(uint32_t, totalWi2h);
    TILING_DATA_FIELD_DEF(uint32_t, totalBi2h);
    TILING_DATA_FIELD_DEF(uint32_t, totalWh2o);
    TILING_DATA_FIELD_DEF(uint32_t, totalBh2o);
    TILING_DATA_FIELD_DEF(uint32_t, totalY);

    TILING_DATA_FIELD_DEF(uint32_t, T);
    TILING_DATA_FIELD_DEF(uint32_t, B);
    TILING_DATA_FIELD_DEF(uint32_t, I);
    TILING_DATA_FIELD_DEF(uint32_t, H);
    TILING_DATA_FIELD_DEF(uint32_t, O);
    TILING_DATA_FIELD_DEF(uint32_t, K);

    TILING_DATA_FIELD_DEF(uint32_t, blockB);
    TILING_DATA_FIELD_DEF(uint32_t, totalB);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(VanillaRnnHiddenCustom, VanillaRnnHiddenCustomTilingData)
} // namespace optiling
