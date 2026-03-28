
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GemmGroupNormHardtanhCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, M);
    TILING_DATA_FIELD_DEF(uint32_t, K);
    TILING_DATA_FIELD_DEF(uint32_t, N);

    TILING_DATA_FIELD_DEF(uint32_t, num_groups);
    TILING_DATA_FIELD_DEF(uint32_t, group_size);

    // total tasks = M * num_groups (each task: one row + one group)
    TILING_DATA_FIELD_DEF(uint32_t, total_tasks);

    // 1 task per block (grid-stride in kernel), capped for occupancy control
    TILING_DATA_FIELD_DEF(uint32_t, block_dim);

    // debug/sanity sizes
    TILING_DATA_FIELD_DEF(uint32_t, total_x);
    TILING_DATA_FIELD_DEF(uint32_t, total_w);
    TILING_DATA_FIELD_DEF(uint32_t, total_bias);
    TILING_DATA_FIELD_DEF(uint32_t, total_gamma);
    TILING_DATA_FIELD_DEF(uint32_t, total_beta);
    TILING_DATA_FIELD_DEF(uint32_t, total_y);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GemmGroupNormHardtanhCustom,
                           GemmGroupNormHardtanhCustomTilingData)

} // namespace optiling
