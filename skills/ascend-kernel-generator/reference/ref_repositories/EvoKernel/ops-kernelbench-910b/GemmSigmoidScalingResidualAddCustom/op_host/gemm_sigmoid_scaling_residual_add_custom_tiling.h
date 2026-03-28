
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GemmSigmoidScalingResidualAddCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalX);
    TILING_DATA_FIELD_DEF(uint32_t, totalW);
    TILING_DATA_FIELD_DEF(uint32_t, totalB);
    TILING_DATA_FIELD_DEF(uint32_t, totalS);
    TILING_DATA_FIELD_DEF(uint32_t, totalY);
    TILING_DATA_FIELD_DEF(uint32_t, M);
    TILING_DATA_FIELD_DEF(uint32_t, K);
    TILING_DATA_FIELD_DEF(uint32_t, N);
    TILING_DATA_FIELD_DEF(uint32_t, totalElems); // M*N
    TILING_DATA_FIELD_DEF(uint32_t, tileElems);  // UB tile size for post ops
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GemmSigmoidScalingResidualAddCustom, GemmSigmoidScalingResidualAddCustomTilingData)
} // namespace optiling
