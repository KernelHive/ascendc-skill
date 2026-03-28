
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(Conv2dAvgPoolSigmoidSumCustomTilingData)
    // x: [N,Cin,H,W]
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, cin);
    TILING_DATA_FIELD_DEF(uint32_t, hin);
    TILING_DATA_FIELD_DEF(uint32_t, win);

    // weight: [Cout,Cin,Kh,Kw]
    TILING_DATA_FIELD_DEF(uint32_t, cout);
    TILING_DATA_FIELD_DEF(uint32_t, kh);
    TILING_DATA_FIELD_DEF(uint32_t, kw);

    // conv out (stride=1,pad=0,dil=1) => [N,Cout,hconv,wconv]
    TILING_DATA_FIELD_DEF(uint32_t, hconv);
    TILING_DATA_FIELD_DEF(uint32_t, wconv);

    // avgpool out (k=4,s=4,p=0,d=1) => [N,Cout,hout,wout]
    TILING_DATA_FIELD_DEF(uint32_t, hout);
    TILING_DATA_FIELD_DEF(uint32_t, wout);

    // y: [N]
    TILING_DATA_FIELD_DEF(uint32_t, total_y);

    // launch geometry (split over N)
    TILING_DATA_FIELD_DEF(uint32_t, block_dim);
    TILING_DATA_FIELD_DEF(uint32_t, n_per_block);

    // debug sizes (elements)
    TILING_DATA_FIELD_DEF(uint32_t, total_x);
    TILING_DATA_FIELD_DEF(uint32_t, total_w);
    TILING_DATA_FIELD_DEF(uint32_t, total_b);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv2dAvgPoolSigmoidSumCustom,
                           Conv2dAvgPoolSigmoidSumCustomTilingData)

} // namespace optiling
