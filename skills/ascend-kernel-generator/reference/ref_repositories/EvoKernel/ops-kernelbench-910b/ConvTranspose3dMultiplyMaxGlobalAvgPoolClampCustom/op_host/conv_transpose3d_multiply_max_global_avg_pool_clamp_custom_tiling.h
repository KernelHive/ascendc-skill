
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ConvTranspose3dMultiplyMaxGlobalAvgPoolClampCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, cin);
    TILING_DATA_FIELD_DEF(uint32_t, din);
    TILING_DATA_FIELD_DEF(uint32_t, hin);
    TILING_DATA_FIELD_DEF(uint32_t, win);

    TILING_DATA_FIELD_DEF(uint32_t, cout);
    TILING_DATA_FIELD_DEF(uint32_t, kd);
    TILING_DATA_FIELD_DEF(uint32_t, kh);
    TILING_DATA_FIELD_DEF(uint32_t, kw);

    TILING_DATA_FIELD_DEF(float, scale);
    TILING_DATA_FIELD_DEF(float, clamp_min);
    TILING_DATA_FIELD_DEF(float, clamp_max);

    // Grouped over output channels: 4 channels per task-group
    TILING_DATA_FIELD_DEF(uint32_t, groups);          // total task-groups = n * (cout/4)
    TILING_DATA_FIELD_DEF(uint32_t, groups_per_block);

    TILING_DATA_FIELD_DEF(uint32_t, total_x);
    TILING_DATA_FIELD_DEF(uint32_t, total_w);
    TILING_DATA_FIELD_DEF(uint32_t, total_b);
    TILING_DATA_FIELD_DEF(uint32_t, total_y);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvTranspose3dMultiplyMaxGlobalAvgPoolClampCustom,
                           ConvTranspose3dMultiplyMaxGlobalAvgPoolClampCustomTilingData)

} // namespace optiling
