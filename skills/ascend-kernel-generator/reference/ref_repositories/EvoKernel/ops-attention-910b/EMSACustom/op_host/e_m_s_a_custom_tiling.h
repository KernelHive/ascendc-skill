
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(EMSACustomTilingData)
    // q: [B,H,NQ,DK], k:[B,H,DK,NK], v:[B,H,NK,DV], y:[B,H,NQ,DV]
    TILING_DATA_FIELD_DEF(uint32_t, B);
    TILING_DATA_FIELD_DEF(uint32_t, H);
    TILING_DATA_FIELD_DEF(uint32_t, NQ);
    TILING_DATA_FIELD_DEF(uint32_t, DK);
    TILING_DATA_FIELD_DEF(uint32_t, NK);
    TILING_DATA_FIELD_DEF(uint32_t, DV);
    TILING_DATA_FIELD_DEF(float, scale);   // 1/sqrt(DK)

    // Parallelization over (B*H): each block loads K/V once and processes all NQ
    TILING_DATA_FIELD_DEF(uint32_t, totalBH);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(EMSACustom, EMSACustomTilingData)
} // namespace optiling
