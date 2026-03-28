
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ParallelPolarizedSelfAttentionCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, B);
    TILING_DATA_FIELD_DEF(uint32_t, C);
    TILING_DATA_FIELD_DEF(uint32_t, H);
    TILING_DATA_FIELD_DEF(uint32_t, W);
    TILING_DATA_FIELD_DEF(uint32_t, HW);
    TILING_DATA_FIELD_DEF(uint32_t, totalPos);     // B*HW
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);     // core num
    TILING_DATA_FIELD_DEF(uint32_t, posPerCore);   // ceil(totalPos/blockDim)
    TILING_DATA_FIELD_DEF(uint32_t, cTile);        // channel tile elems (aligned)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ParallelPolarizedSelfAttentionCustom, ParallelPolarizedSelfAttentionCustomTilingData)
} // namespace optiling
