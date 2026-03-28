
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GlobalFilterCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, B);
    TILING_DATA_FIELD_DEF(uint32_t, N);
    TILING_DATA_FIELD_DEF(uint32_t, C);
    TILING_DATA_FIELD_DEF(uint32_t, H);
    TILING_DATA_FIELD_DEF(uint32_t, W);
    TILING_DATA_FIELD_DEF(uint32_t, FW);          // W/2+1
    TILING_DATA_FIELD_DEF(uint32_t, xTotal);      // B*N*C
    TILING_DATA_FIELD_DEF(uint32_t, wTotal);      // H*FW*C*2
    TILING_DATA_FIELD_DEF(float, invSqrtHW);      // 1/sqrt(H*W) for norm='ortho'
    TILING_DATA_FIELD_DEF(uint32_t, cTile);       // channels per block (tile)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GlobalFilterCustom, GlobalFilterCustomTilingData)
} // namespace optiling
