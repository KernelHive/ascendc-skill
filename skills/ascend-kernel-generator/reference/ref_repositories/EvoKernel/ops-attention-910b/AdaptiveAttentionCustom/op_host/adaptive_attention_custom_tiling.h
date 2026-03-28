
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AdaptiveAttentionCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, B);
    TILING_DATA_FIELD_DEF(uint32_t, H);
    TILING_DATA_FIELD_DEF(uint32_t, S);
    TILING_DATA_FIELD_DEF(uint32_t, D);

    // total rows = B*H*S, one row = one query position for one head
    TILING_DATA_FIELD_DEF(uint32_t, totalRows);

    TILING_DATA_FIELD_DEF(uint32_t, coreNum);
    TILING_DATA_FIELD_DEF(uint32_t, sTile);
    TILING_DATA_FIELD_DEF(uint32_t, dTile);

    TILING_DATA_FIELD_DEF(float, scale); // 1/sqrt(D)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AdaptiveAttentionCustom, AdaptiveAttentionCustomTilingData)
} // namespace optiling
