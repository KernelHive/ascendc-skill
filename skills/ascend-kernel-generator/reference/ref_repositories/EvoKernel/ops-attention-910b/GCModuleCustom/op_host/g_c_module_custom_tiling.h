
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GCModuleCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, N);
    TILING_DATA_FIELD_DEF(uint32_t, C);
    TILING_DATA_FIELD_DEF(uint32_t, H);
    TILING_DATA_FIELD_DEF(uint32_t, W);
    TILING_DATA_FIELD_DEF(uint32_t, HW);
    TILING_DATA_FIELD_DEF(uint32_t, totalElems);   // N*C*H*W
    TILING_DATA_FIELD_DEF(uint32_t, blockElems);   // elems per core (aligned)
    TILING_DATA_FIELD_DEF(uint32_t, tileElems);    // elems per tile (aligned)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GCModuleCustom, GCModuleCustomTilingData)
} // namespace optiling
