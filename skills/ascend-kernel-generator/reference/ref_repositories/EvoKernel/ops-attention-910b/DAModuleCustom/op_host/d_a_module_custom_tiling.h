
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DAModuleCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, bs);
    TILING_DATA_FIELD_DEF(uint32_t, c);
    TILING_DATA_FIELD_DEF(uint32_t, hw);
    TILING_DATA_FIELD_DEF(uint32_t, totalRows);     // bs*hw
    TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);   // ceil(totalRows/blockDim)
    TILING_DATA_FIELD_DEF(uint32_t, cTile);         // channel tile in UB
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DAModuleCustom, DAModuleCustomTilingData)
} // namespace optiling
