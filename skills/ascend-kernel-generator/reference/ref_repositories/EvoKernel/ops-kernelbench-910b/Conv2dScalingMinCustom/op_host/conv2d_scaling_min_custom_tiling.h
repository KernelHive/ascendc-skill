
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(Conv2dScalingMinCustomTilingData)
    // Shapes
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, cin);
    TILING_DATA_FIELD_DEF(uint32_t, hin);
    TILING_DATA_FIELD_DEF(uint32_t, win);

    TILING_DATA_FIELD_DEF(uint32_t, cout);
    TILING_DATA_FIELD_DEF(uint32_t, kh);
    TILING_DATA_FIELD_DEF(uint32_t, kw);

    TILING_DATA_FIELD_DEF(uint32_t, hout);
    TILING_DATA_FIELD_DEF(uint32_t, wout);

    // Specialized scalar (fixed for this benchmark)
    TILING_DATA_FIELD_DEF(float, scale_value);

    // Launch: one block per batch item
    TILING_DATA_FIELD_DEF(uint32_t, blocks);

    // UB tiling for spatial stripe
    TILING_DATA_FIELD_DEF(uint32_t, tile_w);     // number of output width elements per tile (must be even for ow-pair, except tail)
    TILING_DATA_FIELD_DEF(uint32_t, tile_elems); // same as tile_w (for compatibility)

    // Sizes (debug/guard)
    TILING_DATA_FIELD_DEF(uint32_t, total_x);
    TILING_DATA_FIELD_DEF(uint32_t, total_w);
    TILING_DATA_FIELD_DEF(uint32_t, total_b);
    TILING_DATA_FIELD_DEF(uint32_t, total_y);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv2dScalingMinCustom,
                           Conv2dScalingMinCustomTilingData)

} // namespace optiling
