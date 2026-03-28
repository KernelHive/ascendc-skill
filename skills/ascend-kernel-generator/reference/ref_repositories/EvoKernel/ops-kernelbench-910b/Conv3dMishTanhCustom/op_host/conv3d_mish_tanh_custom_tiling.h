
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(Conv3dMishTanhCustomTilingData)
    // flattened element count
    TILING_DATA_FIELD_DEF(uint32_t, total_x);

    // launch geometry
    TILING_DATA_FIELD_DEF(uint32_t, block_dim);
    TILING_DATA_FIELD_DEF(uint32_t, elems_per_block);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv3dMishTanhCustom, Conv3dMishTanhCustomTilingData)

} // namespace optiling
