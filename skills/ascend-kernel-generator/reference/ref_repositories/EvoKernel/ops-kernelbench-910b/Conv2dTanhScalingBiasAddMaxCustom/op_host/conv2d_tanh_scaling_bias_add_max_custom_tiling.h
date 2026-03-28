
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(Conv2dTanhScalingBiasAddMaxCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, cin);
    TILING_DATA_FIELD_DEF(uint32_t, hin);
    TILING_DATA_FIELD_DEF(uint32_t, win);

    TILING_DATA_FIELD_DEF(uint32_t, cout);
    TILING_DATA_FIELD_DEF(uint32_t, kh);
    TILING_DATA_FIELD_DEF(uint32_t, kw);

    TILING_DATA_FIELD_DEF(uint32_t, hout);
    TILING_DATA_FIELD_DEF(uint32_t, wout);

    TILING_DATA_FIELD_DEF(uint32_t, pool_k);
    TILING_DATA_FIELD_DEF(uint32_t, pool_s);
    TILING_DATA_FIELD_DEF(uint32_t, phout);
    TILING_DATA_FIELD_DEF(uint32_t, pwout);

    // Pair mapping and UB tiling
    TILING_DATA_FIELD_DEF(uint32_t, hw_pooled);     // phout*pwout
    TILING_DATA_FIELD_DEF(uint32_t, pairs_total);   // n*cout
    TILING_DATA_FIELD_DEF(uint32_t, tile_hw);       // UB tile along pooled HW

    // Sizes (debug/guard)
    TILING_DATA_FIELD_DEF(uint32_t, total_x);
    TILING_DATA_FIELD_DEF(uint32_t, total_w);
    TILING_DATA_FIELD_DEF(uint32_t, total_conv_bias);
    TILING_DATA_FIELD_DEF(uint32_t, total_post_bias);
    TILING_DATA_FIELD_DEF(uint32_t, total_scaling_factor);
    TILING_DATA_FIELD_DEF(uint32_t, total_y);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv2dTanhScalingBiasAddMaxCustom,
                           Conv2dTanhScalingBiasAddMaxCustomTilingData)

} // namespace optiling
