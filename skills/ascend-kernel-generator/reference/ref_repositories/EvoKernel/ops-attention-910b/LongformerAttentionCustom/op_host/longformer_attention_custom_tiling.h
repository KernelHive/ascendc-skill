
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LongformerAttentionCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, b);
    TILING_DATA_FIELD_DEF(uint32_t, h);
    TILING_DATA_FIELD_DEF(uint32_t, s);
    TILING_DATA_FIELD_DEF(uint32_t, d);
    TILING_DATA_FIELD_DEF(uint32_t, window_size);
    TILING_DATA_FIELD_DEF(uint32_t, max_win);      // 2*(w//2)+1 (interior effective len)
    TILING_DATA_FIELD_DEF(uint32_t, g0);           // fixed global token 0
    TILING_DATA_FIELD_DEF(uint32_t, g1);           // fixed global token 511
    TILING_DATA_FIELD_DEF(float, scale);           // 1/sqrt(d)
    TILING_DATA_FIELD_DEF(uint32_t, total_rows);   // total query rows per (b,h) == s
    TILING_DATA_FIELD_DEF(uint32_t, rows_per_core);// queries computed per core
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LongformerAttentionCustom, LongformerAttentionCustomTilingData)
} // namespace optiling
