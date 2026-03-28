
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(PagedAttentionKVCacheCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, B);
    TILING_DATA_FIELD_DEF(uint32_t, Sq);
    TILING_DATA_FIELD_DEF(uint32_t, Hq);
    TILING_DATA_FIELD_DEF(uint32_t, D);
    TILING_DATA_FIELD_DEF(uint32_t, NB);
    TILING_DATA_FIELD_DEF(uint32_t, PBS);
    TILING_DATA_FIELD_DEF(uint32_t, Hkv);
    TILING_DATA_FIELD_DEF(uint32_t, MBS);
    TILING_DATA_FIELD_DEF(uint32_t, maxSeq);
    TILING_DATA_FIELD_DEF(uint32_t, groups);
    TILING_DATA_FIELD_DEF(uint32_t, totalTasks);
    TILING_DATA_FIELD_DEF(float, scale);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(PagedAttentionKVCacheCustom, PagedAttentionKVCacheCustomTilingData)

} // namespace optiling
