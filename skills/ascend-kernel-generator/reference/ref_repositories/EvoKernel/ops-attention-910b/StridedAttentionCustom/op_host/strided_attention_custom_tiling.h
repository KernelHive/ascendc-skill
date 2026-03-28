
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(StridedAttentionCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, b);
    TILING_DATA_FIELD_DEF(uint32_t, h);
    TILING_DATA_FIELD_DEF(uint32_t, s);
    TILING_DATA_FIELD_DEF(uint32_t, d);
    TILING_DATA_FIELD_DEF(uint32_t, stride);
    TILING_DATA_FIELD_DEF(uint32_t, nsel);      // ceil(s/stride)
    TILING_DATA_FIELD_DEF(float, scale);        // 1/sqrt(d)
    TILING_DATA_FIELD_DEF(uint32_t, totalRows); // b*h*s
    TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(StridedAttentionCustom, StridedAttentionCustomTilingData)
} // namespace optiling
