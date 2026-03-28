
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MatmulMishMishCustomTilingData)
    // Specialized shapes
    TILING_DATA_FIELD_DEF(uint32_t, M);
    TILING_DATA_FIELD_DEF(uint32_t, K);
    TILING_DATA_FIELD_DEF(uint32_t, N);

    // total elements (M*N)
    TILING_DATA_FIELD_DEF(uint32_t, total_elems);

    // Launch geometry
    TILING_DATA_FIELD_DEF(uint32_t, block_dim);
    TILING_DATA_FIELD_DEF(uint32_t, elems_per_block);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatmulMishMishCustom, MatmulMishMishCustomTilingData)

} // namespace optiling
