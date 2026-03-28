
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MatmulSumMaxAvgPoolLogSumExpLogSumExpCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, B);
    TILING_DATA_FIELD_DEF(uint32_t, K);
    TILING_DATA_FIELD_DEF(uint32_t, O);

    // debug/guard sizes
    TILING_DATA_FIELD_DEF(uint32_t, total_x);
    TILING_DATA_FIELD_DEF(uint32_t, total_w);
    TILING_DATA_FIELD_DEF(uint32_t, total_b);
    TILING_DATA_FIELD_DEF(uint32_t, total_y);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatmulSumMaxAvgPoolLogSumExpLogSumExpCustom,
                           MatmulSumMaxAvgPoolLogSumExpLogSumExpCustomTilingData)

} // namespace optiling
