
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(BlockSparseAttentionCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, B);
    TILING_DATA_FIELD_DEF(uint32_t, H);
    TILING_DATA_FIELD_DEF(uint32_t, NB);
    TILING_DATA_FIELD_DEF(uint32_t, BS);
    TILING_DATA_FIELD_DEF(uint32_t, DK);
    TILING_DATA_FIELD_DEF(uint32_t, totalBlocks);   // B*H*NB
    TILING_DATA_FIELD_DEF(uint32_t, coreNum);
    TILING_DATA_FIELD_DEF(uint32_t, dkTile);        // tile for DK
    TILING_DATA_FIELD_DEF(uint32_t, jTile);         // tile for BS in V accumulation
    TILING_DATA_FIELD_DEF(uint32_t, useFastPath32); // 1 when BS==32 and DK%32==0
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BlockSparseAttentionCustom, BlockSparseAttentionCustomTilingData)
} // namespace optiling
