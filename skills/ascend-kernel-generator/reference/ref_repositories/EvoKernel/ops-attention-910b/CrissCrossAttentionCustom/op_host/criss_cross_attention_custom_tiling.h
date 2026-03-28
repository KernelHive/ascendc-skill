
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CrissCrossAttentionCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalElems);
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);
    TILING_DATA_FIELD_DEF(uint32_t, coreElemsAligned);   // per-core aligned vectorizable elements
    TILING_DATA_FIELD_DEF(uint32_t, tileElems);          // UB tile size (aligned)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CrissCrossAttentionCustom, CrissCrossAttentionCustomTilingData)
} // namespace optiling
