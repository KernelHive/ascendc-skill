
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TilingData)
TILING_DATA_FIELD_DEF(uint32_t, rows);
TILING_DATA_FIELD_DEF(uint32_t, cols);
TILING_DATA_FIELD_DEF(uint32_t, totalElems);
TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CumsumCustom, TilingData)
} // namespace optiling
