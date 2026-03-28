
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(Conv2dMinAddMultiplyCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, cin);
    TILING_DATA_FIELD_DEF(uint32_t, hin);
    TILING_DATA_FIELD_DEF(uint32_t, win);

    TILING_DATA_FIELD_DEF(uint32_t, cout);
    TILING_DATA_FIELD_DEF(uint32_t, kh);
    TILING_DATA_FIELD_DEF(uint32_t, kw);

    TILING_DATA_FIELD_DEF(uint32_t, hout);
    TILING_DATA_FIELD_DEF(uint32_t, wout);

    TILING_DATA_FIELD_DEF(float, constant_value);
    TILING_DATA_FIELD_DEF(float, scaling_factor);

    // One block per batch item (stable)
    TILING_DATA_FIELD_DEF(uint32_t, blocks);

    TILING_DATA_FIELD_DEF(uint32_t, total_x);
    TILING_DATA_FIELD_DEF(uint32_t, total_w);
    TILING_DATA_FIELD_DEF(uint32_t, total_conv_bias);
    TILING_DATA_FIELD_DEF(uint32_t, total_post_bias);
    TILING_DATA_FIELD_DEF(uint32_t, total_y);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv2dMinAddMultiplyCustom,
                           Conv2dMinAddMultiplyCustomTilingData)

} // namespace optiling
