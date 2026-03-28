
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(PAMCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, N);
    TILING_DATA_FIELD_DEF(uint32_t, S);
    TILING_DATA_FIELD_DEF(uint32_t, C);
    TILING_DATA_FIELD_DEF(uint32_t, S_pad);        // padded to 64
    TILING_DATA_FIELD_DEF(uint32_t, totalRows);    // N*S
    TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);  // ceil(totalRows/blockDim)
    TILING_DATA_FIELD_DEF(uint32_t, cTile);        // channel tile for UB
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(PAMCustom, PAMCustomTilingData)

} // namespace optiling
