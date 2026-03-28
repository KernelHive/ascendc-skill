
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(Conv3dLeakyReluSumClampGeluCustomTilingData)
    // Shapes
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, cin);
    TILING_DATA_FIELD_DEF(uint32_t, din);
    TILING_DATA_FIELD_DEF(uint32_t, hin);
    TILING_DATA_FIELD_DEF(uint32_t, win);

    TILING_DATA_FIELD_DEF(uint32_t, cout);
    TILING_DATA_FIELD_DEF(uint32_t, kd);
    TILING_DATA_FIELD_DEF(uint32_t, kh);
    TILING_DATA_FIELD_DEF(uint32_t, kw);

    TILING_DATA_FIELD_DEF(uint32_t, dout);
    TILING_DATA_FIELD_DEF(uint32_t, hout);
    TILING_DATA_FIELD_DEF(uint32_t, wout);

    // Conv params (specialized)
    TILING_DATA_FIELD_DEF(uint32_t, stride);
    TILING_DATA_FIELD_DEF(uint32_t, pad);
    TILING_DATA_FIELD_DEF(uint32_t, dilation);
    TILING_DATA_FIELD_DEF(uint32_t, groups);

    // Pointwise scalars
    TILING_DATA_FIELD_DEF(float, leaky_slope);
    TILING_DATA_FIELD_DEF(float, clamp_min);
    TILING_DATA_FIELD_DEF(float, clamp_max);

    // Parallelization (new mapping)
    TILING_DATA_FIELD_DEF(uint32_t, total_tasks);
    TILING_DATA_FIELD_DEF(uint32_t, tasks_per_block);

    // Total sizes (debug/guard)
    TILING_DATA_FIELD_DEF(uint32_t, total_x);
    TILING_DATA_FIELD_DEF(uint32_t, total_w);
    TILING_DATA_FIELD_DEF(uint32_t, total_b);
    TILING_DATA_FIELD_DEF(uint32_t, total_s);
    TILING_DATA_FIELD_DEF(uint32_t, total_y);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv3dLeakyReluSumClampGeluCustom,
                           Conv3dLeakyReluSumClampGeluCustomTilingData)

} // namespace optiling
