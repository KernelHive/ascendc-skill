
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GemmBiasAddHardtanhMishGroupNormCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, M);
    TILING_DATA_FIELD_DEF(uint32_t, K);
    TILING_DATA_FIELD_DEF(uint32_t, N);

    TILING_DATA_FIELD_DEF(uint32_t, num_groups);
    TILING_DATA_FIELD_DEF(uint32_t, group_size);
    TILING_DATA_FIELD_DEF(uint32_t, total_tasks);

    // 1 task per block for higher occupancy and simpler control.
    TILING_DATA_FIELD_DEF(uint32_t, block_dim);

    // debug/sanity
    TILING_DATA_FIELD_DEF(uint32_t, total_x);
    TILING_DATA_FIELD_DEF(uint32_t, total_w);
    TILING_DATA_FIELD_DEF(uint32_t, total_lin_bias);
    TILING_DATA_FIELD_DEF(uint32_t, total_bias);
    TILING_DATA_FIELD_DEF(uint32_t, total_gamma);
    TILING_DATA_FIELD_DEF(uint32_t, total_beta);
    TILING_DATA_FIELD_DEF(uint32_t, total_y);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GemmBiasAddHardtanhMishGroupNormCustom,
                           GemmBiasAddHardtanhMishGroupNormCustomTilingData)

} // namespace optiling
