
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LocalAttentionCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, b);
    TILING_DATA_FIELD_DEF(uint32_t, h);
    TILING_DATA_FIELD_DEF(uint32_t, s);
    TILING_DATA_FIELD_DEF(uint32_t, d);

    TILING_DATA_FIELD_DEF(uint32_t, window_size);
    TILING_DATA_FIELD_DEF(uint32_t, max_win);        // 2*(w//2)+1
    TILING_DATA_FIELD_DEF(float, scale);             // 1/sqrt(d)

    TILING_DATA_FIELD_DEF(uint32_t, total_rows);     // B*H*S
    TILING_DATA_FIELD_DEF(uint32_t, rows_per_core);  // ceil(total_rows / block_dim)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LocalAttentionCustom, LocalAttentionCustomTilingData)
} // namespace optiling
