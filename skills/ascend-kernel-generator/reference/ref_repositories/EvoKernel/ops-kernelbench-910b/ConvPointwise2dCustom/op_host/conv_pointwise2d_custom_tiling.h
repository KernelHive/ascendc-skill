
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvPointwise2dCustomTilingData)
    // Specialized contract:
    // x:      [16, 64, 1024, 1024]
    // weight: [128, 64, 1, 1] (OIHW)
    // y:      [16, 128, 1024, 1024]
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, cin);
    TILING_DATA_FIELD_DEF(uint32_t, cout);
    TILING_DATA_FIELD_DEF(uint32_t, h);
    TILING_DATA_FIELD_DEF(uint32_t, w);

    // Task mapping:
    // task -> (nIdx, hIdx, wGroupIdx)
    // where wGroupIdx covers WGROUP*WTILE contiguous elements.
    TILING_DATA_FIELD_DEF(uint32_t, w_tile);      // WTILE
    TILING_DATA_FIELD_DEF(uint32_t, w_group);     // WGROUP
    TILING_DATA_FIELD_DEF(uint32_t, w_group_len); // WTILE*WGROUP
    TILING_DATA_FIELD_DEF(uint32_t, w_groups);    // ceil(w / (WTILE*WGROUP))
    TILING_DATA_FIELD_DEF(uint32_t, tasks);       // n*h*w_groups
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvPointwise2dCustom, ConvPointwise2dCustomTilingData)
} // namespace optiling
