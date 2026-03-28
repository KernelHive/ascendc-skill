
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(UnPermuteCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, m);        // num_tokens
    TILING_DATA_FIELD_DEF(uint32_t, k);        // hidden_size
    TILING_DATA_FIELD_DEF(uint32_t, topk);     // topk
    TILING_DATA_FIELD_DEF(uint32_t, kTile);    // tile along K
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(UnPermuteCustom, UnPermuteCustomTilingData)
} // namespace optiling
