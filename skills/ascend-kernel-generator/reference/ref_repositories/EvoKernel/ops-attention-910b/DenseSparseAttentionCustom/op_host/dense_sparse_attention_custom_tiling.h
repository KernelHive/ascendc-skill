
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(DenseSparseAttentionCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, B);
    TILING_DATA_FIELD_DEF(uint32_t, Sq);
    TILING_DATA_FIELD_DEF(uint32_t, H);
    TILING_DATA_FIELD_DEF(uint32_t, Dqk);
    TILING_DATA_FIELD_DEF(uint32_t, Dv);
    TILING_DATA_FIELD_DEF(uint32_t, NB);
    TILING_DATA_FIELD_DEF(uint32_t, PBS);
    TILING_DATA_FIELD_DEF(uint32_t, topk);
    TILING_DATA_FIELD_DEF(uint32_t, flatKV);
    TILING_DATA_FIELD_DEF(uint32_t, totalTasks);
    TILING_DATA_FIELD_DEF(uint32_t, coreNum);
    TILING_DATA_FIELD_DEF(float, scale);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DenseSparseAttentionCustom, DenseSparseAttentionCustomTilingData)

} // namespace optiling
