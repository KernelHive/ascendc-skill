
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CoTAttentionCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, bs);
    TILING_DATA_FIELD_DEF(uint32_t, C);
    TILING_DATA_FIELD_DEF(uint32_t, H);
    TILING_DATA_FIELD_DEF(uint32_t, W);
    TILING_DATA_FIELD_DEF(uint32_t, hw);
    TILING_DATA_FIELD_DEF(uint32_t, totalRows);       // bs*C
    TILING_DATA_FIELD_DEF(uint32_t, blockRows);       // rows handled per core (already includes unroll)
    TILING_DATA_FIELD_DEF(uint32_t, unroll);          // =2
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CoTAttentionCustom, CoTAttentionCustomTilingData)
} // namespace optiling
