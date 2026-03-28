
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AxialAttentionCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, bh);   // B*H (merged batch-head)
    TILING_DATA_FIELD_DEF(uint32_t, t);    // sequence length along axis
    TILING_DATA_FIELD_DEF(uint32_t, e);    // head dim
    TILING_DATA_FIELD_DEF(float, scale);  // scaling factor
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AxialAttentionCustom, AxialAttentionCustomTilingData)
} // namespace optiling
