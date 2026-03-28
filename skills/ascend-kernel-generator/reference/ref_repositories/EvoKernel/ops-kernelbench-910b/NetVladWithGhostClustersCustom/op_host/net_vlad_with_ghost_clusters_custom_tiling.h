
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(NetVladWithGhostClustersCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalX);
    TILING_DATA_FIELD_DEF(uint32_t, totalClusters);
    TILING_DATA_FIELD_DEF(uint32_t, totalClusters2);
    TILING_DATA_FIELD_DEF(uint32_t, totalBnW);
    TILING_DATA_FIELD_DEF(uint32_t, totalBnB);
    TILING_DATA_FIELD_DEF(uint32_t, totalBnM);
    TILING_DATA_FIELD_DEF(uint32_t, totalBnV);
    TILING_DATA_FIELD_DEF(uint32_t, totalY);

    TILING_DATA_FIELD_DEF(uint32_t, B);
    TILING_DATA_FIELD_DEF(uint32_t, N);
    TILING_DATA_FIELD_DEF(uint32_t, D);
    TILING_DATA_FIELD_DEF(uint32_t, K);
    TILING_DATA_FIELD_DEF(uint32_t, Kall);

    TILING_DATA_FIELD_DEF(float, bnEps);
    TILING_DATA_FIELD_DEF(float, l2Eps);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(NetVladWithGhostClustersCustom,
                           NetVladWithGhostClustersCustomTilingData)
} // namespace optiling
