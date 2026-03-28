
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ConvTranspose3dMaxPoolSoftmaxSubtractSwishMaxCustomTilingData)
    // x: [N,Cin,D,H,W]
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, cin);
    TILING_DATA_FIELD_DEF(uint32_t, din);
    TILING_DATA_FIELD_DEF(uint32_t, hin);
    TILING_DATA_FIELD_DEF(uint32_t, win);

    // weight: [Cin,Cout,Kd,Kh,Kw]
    TILING_DATA_FIELD_DEF(uint32_t, cout);
    TILING_DATA_FIELD_DEF(uint32_t, kd);
    TILING_DATA_FIELD_DEF(uint32_t, kh);
    TILING_DATA_FIELD_DEF(uint32_t, kw);

    // derived sizes
    TILING_DATA_FIELD_DEF(uint32_t, dout);
    TILING_DATA_FIELD_DEF(uint32_t, hout);
    TILING_DATA_FIELD_DEF(uint32_t, wout);

    TILING_DATA_FIELD_DEF(uint32_t, dp);
    TILING_DATA_FIELD_DEF(uint32_t, hp);
    TILING_DATA_FIELD_DEF(uint32_t, wp);

    // linearized output size
    TILING_DATA_FIELD_DEF(uint32_t, total_y);

    // launch geometry helpers
    TILING_DATA_FIELD_DEF(uint32_t, block_dim);
    TILING_DATA_FIELD_DEF(uint32_t, elems_per_block);

    // debug/guard sizes (elements)
    TILING_DATA_FIELD_DEF(uint32_t, total_x);
    TILING_DATA_FIELD_DEF(uint32_t, total_w);
    TILING_DATA_FIELD_DEF(uint32_t, total_conv_b);
    TILING_DATA_FIELD_DEF(uint32_t, total_sub);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvTranspose3dMaxPoolSoftmaxSubtractSwishMaxCustom,
                           ConvTranspose3dMaxPoolSoftmaxSubtractSwishMaxCustomTilingData)

} // namespace optiling
