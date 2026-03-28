
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GemmMaxSubtractGeluCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalX);
    TILING_DATA_FIELD_DEF(uint32_t, totalW);
    TILING_DATA_FIELD_DEF(uint32_t, totalB);
    TILING_DATA_FIELD_DEF(uint32_t, totalY);
    TILING_DATA_FIELD_DEF(uint32_t, M);
    TILING_DATA_FIELD_DEF(uint32_t, K);
    TILING_DATA_FIELD_DEF(uint32_t, N);
    TILING_DATA_FIELD_DEF(uint32_t, totalOutElems); // M*1
    TILING_DATA_FIELD_DEF(uint32_t, tileElems);     // UB tile for vectorized zero-fill
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GemmMaxSubtractGeluCustom, GemmMaxSubtractGeluCustomTilingData)
} // namespace optiling
