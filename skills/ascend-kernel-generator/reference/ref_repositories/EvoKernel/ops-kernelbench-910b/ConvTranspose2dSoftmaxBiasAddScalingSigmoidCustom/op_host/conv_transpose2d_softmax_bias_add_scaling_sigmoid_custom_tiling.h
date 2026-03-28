
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ConvTranspose2dSoftmaxBiasAddScalingSigmoidCustomTilingData)
    // Shapes
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, cin);
    TILING_DATA_FIELD_DEF(uint32_t, hin);
    TILING_DATA_FIELD_DEF(uint32_t, win);

    TILING_DATA_FIELD_DEF(uint32_t, cout);
    TILING_DATA_FIELD_DEF(uint32_t, kh);
    TILING_DATA_FIELD_DEF(uint32_t, kw);

    // ConvTranspose2d attributes
    TILING_DATA_FIELD_DEF(uint32_t, stride);
    TILING_DATA_FIELD_DEF(uint32_t, pad);
    TILING_DATA_FIELD_DEF(uint32_t, out_pad);
    TILING_DATA_FIELD_DEF(uint32_t, dilation);

    // Output spatial
    TILING_DATA_FIELD_DEF(uint32_t, hout);
    TILING_DATA_FIELD_DEF(uint32_t, wout);

    // Fused constants
    TILING_DATA_FIELD_DEF(float, scaling);

    // Parallelization
    TILING_DATA_FIELD_DEF(uint32_t, blocks);

    // Sizes (debug/guard)
    TILING_DATA_FIELD_DEF(uint32_t, total_x);
    TILING_DATA_FIELD_DEF(uint32_t, total_w);
    TILING_DATA_FIELD_DEF(uint32_t, total_conv_bias);
    TILING_DATA_FIELD_DEF(uint32_t, total_bias);
    TILING_DATA_FIELD_DEF(uint32_t, total_y);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvTranspose2dSoftmaxBiasAddScalingSigmoidCustom,
                           ConvTranspose2dSoftmaxBiasAddScalingSigmoidCustomTilingData)

} // namespace optiling
