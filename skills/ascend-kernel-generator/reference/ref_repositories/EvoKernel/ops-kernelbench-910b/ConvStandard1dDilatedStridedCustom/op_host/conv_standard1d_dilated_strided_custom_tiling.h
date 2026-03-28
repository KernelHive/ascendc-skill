
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvStandard1dDilatedStridedCustomTilingData)
    // One block per (n, co), each block iterates over lChunks.
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, cin);
    TILING_DATA_FIELD_DEF(uint32_t, cout);
    TILING_DATA_FIELD_DEF(uint32_t, lin);
    TILING_DATA_FIELD_DEF(uint32_t, lout);

    // Chunked sweep of Lout inside each (n,co) block.
    TILING_DATA_FIELD_DEF(uint32_t, chunk_lout);
    TILING_DATA_FIELD_DEF(uint32_t, lout_chunks);

    // Total tasks = N * COUT * lout_chunks (used only for safety checks/debug).
    TILING_DATA_FIELD_DEF(uint32_t, tasks);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvStandard1dDilatedStridedCustom,
                           ConvStandard1dDilatedStridedCustomTilingData)
} // namespace optiling
