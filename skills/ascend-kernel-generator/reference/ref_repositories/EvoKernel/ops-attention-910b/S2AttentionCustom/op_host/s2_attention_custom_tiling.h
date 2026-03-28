
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(S2AttentionCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, B);
    TILING_DATA_FIELD_DEF(uint32_t, K);          // must be 3
    TILING_DATA_FIELD_DEF(uint32_t, H);
    TILING_DATA_FIELD_DEF(uint32_t, W);
    TILING_DATA_FIELD_DEF(uint32_t, C);

    TILING_DATA_FIELD_DEF(uint32_t, HW);         // H*W
    TILING_DATA_FIELD_DEF(uint32_t, rows);       // B*HW
    TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);

    TILING_DATA_FIELD_DEF(uint32_t, cTile);      // channel tile (aligned)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(S2AttentionCustom, S2AttentionCustomTilingData)
} // namespace optiling
