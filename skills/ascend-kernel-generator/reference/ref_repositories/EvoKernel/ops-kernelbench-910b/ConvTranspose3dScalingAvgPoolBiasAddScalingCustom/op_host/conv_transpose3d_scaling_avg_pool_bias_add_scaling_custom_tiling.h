
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ConvTranspose3dScalingAvgPoolBiasAddScalingCustomTilingData)
    // Shapes
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, cin);
    TILING_DATA_FIELD_DEF(uint32_t, din);
    TILING_DATA_FIELD_DEF(uint32_t, hin);
    TILING_DATA_FIELD_DEF(uint32_t, win);

    TILING_DATA_FIELD_DEF(uint32_t, cout);
    TILING_DATA_FIELD_DEF(uint32_t, kd);
    TILING_DATA_FIELD_DEF(uint32_t, kh);
    TILING_DATA_FIELD_DEF(uint32_t, kw);

    // Specialized constants
    TILING_DATA_FIELD_DEF(float, scale1);
    TILING_DATA_FIELD_DEF(float, scale2);

    // Output sizes (avgpool result)
    TILING_DATA_FIELD_DEF(uint32_t, dp);
    TILING_DATA_FIELD_DEF(uint32_t, hp);
    TILING_DATA_FIELD_DEF(uint32_t, wp);

    // Parallelization: one block per batch item
    TILING_DATA_FIELD_DEF(uint32_t, blocks);

    // Sizes (debug/guard)
    TILING_DATA_FIELD_DEF(uint32_t, total_x);
    TILING_DATA_FIELD_DEF(uint32_t, total_w);
    TILING_DATA_FIELD_DEF(uint32_t, total_conv_bias);
    TILING_DATA_FIELD_DEF(uint32_t, total_bias);
    TILING_DATA_FIELD_DEF(uint32_t, total_y);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvTranspose3dScalingAvgPoolBiasAddScalingCustom,
                           ConvTranspose3dScalingAvgPoolBiasAddScalingCustomTilingData)

} // namespace optiling
