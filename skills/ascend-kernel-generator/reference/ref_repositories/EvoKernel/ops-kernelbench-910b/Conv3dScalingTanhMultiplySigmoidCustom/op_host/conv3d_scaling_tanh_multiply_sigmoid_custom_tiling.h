
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(Conv3dScalingTanhMultiplySigmoidCustomTilingData)
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

    // Output sizes
    TILING_DATA_FIELD_DEF(uint32_t, dout);
    TILING_DATA_FIELD_DEF(uint32_t, hout);
    TILING_DATA_FIELD_DEF(uint32_t, wout);

    // Conv params (specialized)
    TILING_DATA_FIELD_DEF(uint32_t, stride);
    TILING_DATA_FIELD_DEF(uint32_t, pad);
    TILING_DATA_FIELD_DEF(uint32_t, dilation);
    TILING_DATA_FIELD_DEF(uint32_t, groups);

    // Parallelization: 2D grid flattened into 1D blockIdx:
    // block = n * cout + co
    TILING_DATA_FIELD_DEF(uint32_t, blocks);

    // Inner tiling over spatial (DHW)
    TILING_DATA_FIELD_DEF(uint32_t, dhw);        // dout*hout*wout
    TILING_DATA_FIELD_DEF(uint32_t, tile);       // vector tile length

    // Sizes (guard/debug)
    TILING_DATA_FIELD_DEF(uint32_t, total_x);
    TILING_DATA_FIELD_DEF(uint32_t, total_w);
    TILING_DATA_FIELD_DEF(uint32_t, total_conv_bias);
    TILING_DATA_FIELD_DEF(uint32_t, total_scaling);
    TILING_DATA_FIELD_DEF(uint32_t, total_bias);
    TILING_DATA_FIELD_DEF(uint32_t, total_y);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv3dScalingTanhMultiplySigmoidCustom,
                           Conv3dScalingTanhMultiplySigmoidCustomTilingData)

} // namespace optiling
