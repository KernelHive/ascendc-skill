
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DoubleAttentionCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, B);
    TILING_DATA_FIELD_DEF(uint32_t, Cm);
    TILING_DATA_FIELD_DEF(uint32_t, Cn);
    TILING_DATA_FIELD_DEF(uint32_t, HW);
    TILING_DATA_FIELD_DEF(uint32_t, batchesPerCore); // ceil(B/blockDim)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DoubleAttentionCustom, DoubleAttentionCustomTilingData)
} // namespace optiling
