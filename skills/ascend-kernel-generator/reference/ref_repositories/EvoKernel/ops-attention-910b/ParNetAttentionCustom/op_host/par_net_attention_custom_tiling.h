
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ParNetAttentionCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalElems);
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);
    TILING_DATA_FIELD_DEF(uint32_t, elemsPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, tileElems);        // elements per tile (aligned)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ParNetAttentionCustom, ParNetAttentionCustomTilingData)

} // namespace optiling
