
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatmulDivideGeluCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalX);
    TILING_DATA_FIELD_DEF(uint32_t, totalW);
    TILING_DATA_FIELD_DEF(uint32_t, totalB);
    TILING_DATA_FIELD_DEF(uint32_t, totalDivisor);
    TILING_DATA_FIELD_DEF(uint32_t, totalY);

    TILING_DATA_FIELD_DEF(uint32_t, M);
    TILING_DATA_FIELD_DEF(uint32_t, K);
    TILING_DATA_FIELD_DEF(uint32_t, N);

    // 2D mapping: each block computes one row and a tile of columns.
    TILING_DATA_FIELD_DEF(uint32_t, nTile);

    // Micro-vectorization within a tile (kernel supports 1/2/4).
    TILING_DATA_FIELD_DEF(uint32_t, vecN);

    // GELU vector tile size for UB processing (small, fixed).
    TILING_DATA_FIELD_DEF(uint32_t, geluTile);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatmulDivideGeluCustom, MatmulDivideGeluCustomTilingData)
} // namespace optiling
