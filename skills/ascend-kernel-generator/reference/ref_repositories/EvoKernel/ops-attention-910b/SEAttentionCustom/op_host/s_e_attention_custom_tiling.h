
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SEAttentionCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, B);
    TILING_DATA_FIELD_DEF(uint32_t, C);
    TILING_DATA_FIELD_DEF(uint32_t, H);
    TILING_DATA_FIELD_DEF(uint32_t, W);
    TILING_DATA_FIELD_DEF(uint32_t, R);
    TILING_DATA_FIELD_DEF(uint32_t, HW);
    TILING_DATA_FIELD_DEF(uint32_t, cTile);
    TILING_DATA_FIELD_DEF(uint32_t, rAlign);
    TILING_DATA_FIELD_DEF(float, invHW);
    TILING_DATA_FIELD_DEF(uint32_t, sigTmpBytes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SEAttentionCustom, SEAttentionCustomTilingData)
} // namespace optiling
