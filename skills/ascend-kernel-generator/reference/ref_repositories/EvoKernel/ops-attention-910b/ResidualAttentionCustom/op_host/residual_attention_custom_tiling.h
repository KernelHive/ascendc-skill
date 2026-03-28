
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ResidualAttentionCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, B);
    TILING_DATA_FIELD_DEF(uint32_t, C);
    TILING_DATA_FIELD_DEF(uint32_t, H);
    TILING_DATA_FIELD_DEF(uint32_t, W);
    TILING_DATA_FIELD_DEF(uint32_t, HW);
    TILING_DATA_FIELD_DEF(uint32_t, totalRows);     // B*C
    TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);   // ceil(totalRows/blockDim)
    TILING_DATA_FIELD_DEF(float, invHW);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ResidualAttentionCustom, ResidualAttentionCustomTilingData)
} // namespace optiling
