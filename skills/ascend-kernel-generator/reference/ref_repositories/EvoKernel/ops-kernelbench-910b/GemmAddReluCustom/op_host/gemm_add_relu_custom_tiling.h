
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GemmAddReluCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalX);
    TILING_DATA_FIELD_DEF(uint32_t, totalW);
    TILING_DATA_FIELD_DEF(uint32_t, totalB);
    TILING_DATA_FIELD_DEF(uint32_t, totalY);

    TILING_DATA_FIELD_DEF(uint32_t, M);
    TILING_DATA_FIELD_DEF(uint32_t, K);
    TILING_DATA_FIELD_DEF(uint32_t, N);

    // Row-parallel scheduling
    TILING_DATA_FIELD_DEF(uint32_t, rowsPerBlock);

    // Micro-tile along N: compute 2 columns at once to reuse X loads
    TILING_DATA_FIELD_DEF(uint32_t, vecN);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GemmAddReluCustom, GemmAddReluCustomTilingData)
} // namespace optiling
