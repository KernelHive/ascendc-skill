
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CrossformerAttentionCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, B);
    TILING_DATA_FIELD_DEF(uint32_t, H);
    TILING_DATA_FIELD_DEF(uint32_t, N);
    TILING_DATA_FIELD_DEF(uint32_t, Dh);
    TILING_DATA_FIELD_DEF(uint32_t, C);            // H*Dh
    TILING_DATA_FIELD_DEF(uint32_t, totalRows);    // B*N
    TILING_DATA_FIELD_DEF(uint32_t, blockRows);    // rows per core
    TILING_DATA_FIELD_DEF(uint32_t, dhTile);       // Dh tile in UB
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CrossformerAttentionCustom, CrossformerAttentionCustomTilingData)
} // namespace optiling
