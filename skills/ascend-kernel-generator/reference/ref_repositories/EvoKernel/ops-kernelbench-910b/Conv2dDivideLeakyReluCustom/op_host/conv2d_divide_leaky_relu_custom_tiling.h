
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(Conv2dDivideLeakyReluCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, cin);
    TILING_DATA_FIELD_DEF(uint32_t, hin);
    TILING_DATA_FIELD_DEF(uint32_t, win);

    TILING_DATA_FIELD_DEF(uint32_t, cout);
    TILING_DATA_FIELD_DEF(uint32_t, kh);
    TILING_DATA_FIELD_DEF(uint32_t, kw);

    TILING_DATA_FIELD_DEF(uint32_t, hout);
    TILING_DATA_FIELD_DEF(uint32_t, wout);

    TILING_DATA_FIELD_DEF(uint32_t, stride);
    TILING_DATA_FIELD_DEF(uint32_t, pad);
    TILING_DATA_FIELD_DEF(uint32_t, dilation);
    TILING_DATA_FIELD_DEF(uint32_t, groups);

    TILING_DATA_FIELD_DEF(float, inv_divisor);
    TILING_DATA_FIELD_DEF(float, leaky_slope);

    // task = (n, co, oh_pair) where oh0 = 2*oh_pair
    TILING_DATA_FIELD_DEF(uint32_t, total_pair_tasks);
    TILING_DATA_FIELD_DEF(uint32_t, pairs_per_block);

    // Precomputed products
    TILING_DATA_FIELD_DEF(uint32_t, hw_out);      // hout*wout
    TILING_DATA_FIELD_DEF(uint32_t, cout_hw_out); // cout*hw_out
    TILING_DATA_FIELD_DEF(uint32_t, cin_hw_in);   // cin*hin*win
    TILING_DATA_FIELD_DEF(uint32_t, hw_in);       // hin*win

    // Debug sizes
    TILING_DATA_FIELD_DEF(uint32_t, total_x);
    TILING_DATA_FIELD_DEF(uint32_t, total_w);
    TILING_DATA_FIELD_DEF(uint32_t, total_b);
    TILING_DATA_FIELD_DEF(uint32_t, total_y);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv2dDivideLeakyReluCustom,
                           Conv2dDivideLeakyReluCustomTilingData)

} // namespace optiling
