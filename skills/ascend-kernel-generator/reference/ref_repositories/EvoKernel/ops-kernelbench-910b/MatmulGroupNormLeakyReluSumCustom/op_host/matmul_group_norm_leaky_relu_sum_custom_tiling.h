
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatmulGroupNormLeakyReluSumCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalX);
    TILING_DATA_FIELD_DEF(uint32_t, totalW);
    TILING_DATA_FIELD_DEF(uint32_t, totalBias);
    TILING_DATA_FIELD_DEF(uint32_t, totalGamma);
    TILING_DATA_FIELD_DEF(uint32_t, totalBeta);
    TILING_DATA_FIELD_DEF(uint32_t, totalEps);
    TILING_DATA_FIELD_DEF(uint32_t, totalNegativeSlope);
    TILING_DATA_FIELD_DEF(uint32_t, totalY);

    TILING_DATA_FIELD_DEF(uint32_t, M);
    TILING_DATA_FIELD_DEF(uint32_t, K);
    TILING_DATA_FIELD_DEF(uint32_t, N);
    TILING_DATA_FIELD_DEF(uint32_t, G);
    TILING_DATA_FIELD_DEF(uint32_t, groupSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatmulGroupNormLeakyReluSumCustom, MatmulGroupNormLeakyReluSumCustomTilingData)
} // namespace optiling
