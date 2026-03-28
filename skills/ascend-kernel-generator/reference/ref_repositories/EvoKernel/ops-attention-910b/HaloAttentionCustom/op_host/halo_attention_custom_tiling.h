
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(HaloAttentionCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, B);          // B'
    TILING_DATA_FIELD_DEF(uint32_t, I);          // query length
    TILING_DATA_FIELD_DEF(uint32_t, J);          // key/value length
    TILING_DATA_FIELD_DEF(uint32_t, D);          // head dim
    TILING_DATA_FIELD_DEF(uint32_t, totalRows);  // B'*I
    TILING_DATA_FIELD_DEF(uint32_t, coreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(HaloAttentionCustom, HaloAttentionCustomTilingData)
} // namespace optiling
