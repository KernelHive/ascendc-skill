
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(Conv3dDivideMaxGlobalAvgPoolBiasAddSumCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, cin);
    TILING_DATA_FIELD_DEF(uint32_t, din);
    TILING_DATA_FIELD_DEF(uint32_t, hin);
    TILING_DATA_FIELD_DEF(uint32_t, win);

    TILING_DATA_FIELD_DEF(uint32_t, cout);
    TILING_DATA_FIELD_DEF(uint32_t, kd);
    TILING_DATA_FIELD_DEF(uint32_t, kh);
    TILING_DATA_FIELD_DEF(uint32_t, kw);

    TILING_DATA_FIELD_DEF(uint32_t, dout);
    TILING_DATA_FIELD_DEF(uint32_t, hout);
    TILING_DATA_FIELD_DEF(uint32_t, wout);

    TILING_DATA_FIELD_DEF(uint32_t, dp);
    TILING_DATA_FIELD_DEF(uint32_t, hp);
    TILING_DATA_FIELD_DEF(uint32_t, wp);

    TILING_DATA_FIELD_DEF(uint32_t, blocks);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv3dDivideMaxGlobalAvgPoolBiasAddSumCustom,
                           Conv3dDivideMaxGlobalAvgPoolBiasAddSumCustomTilingData)

} // namespace optiling
