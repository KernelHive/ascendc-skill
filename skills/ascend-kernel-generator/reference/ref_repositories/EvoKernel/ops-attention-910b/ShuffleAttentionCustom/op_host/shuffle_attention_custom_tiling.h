
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ShuffleAttentionCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, B);
    TILING_DATA_FIELD_DEF(uint32_t, C);      // full channels
    TILING_DATA_FIELD_DEF(uint32_t, H);
    TILING_DATA_FIELD_DEF(uint32_t, W);
    TILING_DATA_FIELD_DEF(uint32_t, G);      // groups in the module (e.g., 8)
    TILING_DATA_FIELD_DEF(uint32_t, C2g);    // C/(2*G)
    TILING_DATA_FIELD_DEF(uint32_t, HW);
    TILING_DATA_FIELD_DEF(uint32_t, yElems); // B*C*H*W

    TILING_DATA_FIELD_DEF(uint32_t, coreNum);
    TILING_DATA_FIELD_DEF(uint32_t, tileNum);
    TILING_DATA_FIELD_DEF(uint32_t, tileElems);
    TILING_DATA_FIELD_DEF(uint32_t, lastTileElems);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ShuffleAttentionCustom, ShuffleAttentionCustomTilingData)
} // namespace optiling
