
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(BAMCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalElems);
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);
    TILING_DATA_FIELD_DEF(uint32_t, elemsPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, tileElems);        // aligned to 64 elems
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BAMCustom, BAMCustomTilingData)

} // namespace optiling
