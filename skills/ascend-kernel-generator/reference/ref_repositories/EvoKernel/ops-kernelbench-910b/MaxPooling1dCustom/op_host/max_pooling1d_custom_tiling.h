
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MaxPooling1dCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, c);
    TILING_DATA_FIELD_DEF(uint32_t, l_in);
    TILING_DATA_FIELD_DEF(uint32_t, l_out);

    TILING_DATA_FIELD_DEF(uint32_t, totalRows);     // N*C
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);      // blocks launched
    TILING_DATA_FIELD_DEF(uint32_t, rowsPerBlock);  // contiguous rows per block
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MaxPooling1dCustom, MaxPooling1dCustomTilingData)
} // namespace optiling
