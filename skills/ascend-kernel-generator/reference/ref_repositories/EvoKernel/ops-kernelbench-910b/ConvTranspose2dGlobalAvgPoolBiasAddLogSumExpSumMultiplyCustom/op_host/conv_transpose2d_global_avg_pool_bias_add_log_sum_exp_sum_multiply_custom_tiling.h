
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ConvTranspose2dGlobalAvgPoolBiasAddLogSumExpSumMultiplyCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, cin);
    TILING_DATA_FIELD_DEF(uint32_t, hin);
    TILING_DATA_FIELD_DEF(uint32_t, win);

    TILING_DATA_FIELD_DEF(uint32_t, cout);

    TILING_DATA_FIELD_DEF(uint32_t, hout);
    TILING_DATA_FIELD_DEF(uint32_t, wout);

    TILING_DATA_FIELD_DEF(float, mul);
    TILING_DATA_FIELD_DEF(float, inv_hw_out);

    TILING_DATA_FIELD_DEF(uint32_t, total_x);
    TILING_DATA_FIELD_DEF(uint32_t, total_sumw);
    TILING_DATA_FIELD_DEF(uint32_t, total_conv_b);
    TILING_DATA_FIELD_DEF(uint32_t, total_b);
    TILING_DATA_FIELD_DEF(uint32_t, total_y);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvTranspose2dGlobalAvgPoolBiasAddLogSumExpSumMultiplyCustom,
                           ConvTranspose2dGlobalAvgPoolBiasAddLogSumExpSumMultiplyCustomTilingData)

} // namespace optiling
