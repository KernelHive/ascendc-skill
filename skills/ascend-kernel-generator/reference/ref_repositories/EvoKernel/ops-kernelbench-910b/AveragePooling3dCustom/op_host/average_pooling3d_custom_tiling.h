
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AveragePooling3dCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, c);
    TILING_DATA_FIELD_DEF(uint32_t, d_in);
    TILING_DATA_FIELD_DEF(uint32_t, h_in);
    TILING_DATA_FIELD_DEF(uint32_t, w_in);

    TILING_DATA_FIELD_DEF(uint32_t, d_out);
    TILING_DATA_FIELD_DEF(uint32_t, h_out);
    TILING_DATA_FIELD_DEF(uint32_t, w_out);

    TILING_DATA_FIELD_DEF(uint32_t, rows);
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);
    TILING_DATA_FIELD_DEF(uint32_t, rowsPerBlock);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AveragePooling3dCustom, AveragePooling3dCustomTilingData)
} // namespace optiling
