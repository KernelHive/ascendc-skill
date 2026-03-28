
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GemmGroupNormSwishMultiplySwishCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalX);
    TILING_DATA_FIELD_DEF(uint32_t, totalW);
    TILING_DATA_FIELD_DEF(uint32_t, totalB);
    TILING_DATA_FIELD_DEF(uint32_t, totalGamma);
    TILING_DATA_FIELD_DEF(uint32_t, totalBeta);
    TILING_DATA_FIELD_DEF(uint32_t, totalMulW);
    TILING_DATA_FIELD_DEF(uint32_t, totalNumGroups);
    TILING_DATA_FIELD_DEF(uint32_t, totalEps);
    TILING_DATA_FIELD_DEF(uint32_t, totalY);

    TILING_DATA_FIELD_DEF(uint32_t, M);
    TILING_DATA_FIELD_DEF(uint32_t, K);
    TILING_DATA_FIELD_DEF(uint32_t, N);
    TILING_DATA_FIELD_DEF(uint32_t, totalElems); // M*N

    TILING_DATA_FIELD_DEF(uint32_t, G);
    TILING_DATA_FIELD_DEF(uint32_t, groupSize); // N/G
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GemmGroupNormSwishMultiplySwishCustom,
                           GemmGroupNormSwishMultiplySwishCustomTilingData)
} // namespace optiling
