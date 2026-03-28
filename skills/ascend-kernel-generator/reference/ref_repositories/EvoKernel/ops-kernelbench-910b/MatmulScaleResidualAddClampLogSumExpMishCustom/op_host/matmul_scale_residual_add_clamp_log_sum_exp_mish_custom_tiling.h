
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MatmulScaleResidualAddClampLogSumExpMishCustomTilingData)
    // Shapes
    TILING_DATA_FIELD_DEF(uint32_t, M);
    TILING_DATA_FIELD_DEF(uint32_t, K);
    TILING_DATA_FIELD_DEF(uint32_t, N);

    // Linearized sizes (elements)
    TILING_DATA_FIELD_DEF(uint32_t, total_x);
    TILING_DATA_FIELD_DEF(uint32_t, total_w);
    TILING_DATA_FIELD_DEF(uint32_t, total_b);
    TILING_DATA_FIELD_DEF(uint32_t, total_y);

    // Launch geometry
    TILING_DATA_FIELD_DEF(uint32_t, block_dim);
    TILING_DATA_FIELD_DEF(uint32_t, rows_per_block);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatmulScaleResidualAddClampLogSumExpMishCustom,
                           MatmulScaleResidualAddClampLogSumExpMishCustomTilingData)

} // namespace optiling
