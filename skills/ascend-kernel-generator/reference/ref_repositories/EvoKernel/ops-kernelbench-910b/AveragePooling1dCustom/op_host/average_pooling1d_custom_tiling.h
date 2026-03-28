
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AveragePooling1dCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalX);   // N*C*L
    TILING_DATA_FIELD_DEF(uint32_t, totalY);   // N*C*Lout
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, c);
    TILING_DATA_FIELD_DEF(uint32_t, l_in);
    TILING_DATA_FIELD_DEF(uint32_t, l_out);
    TILING_DATA_FIELD_DEF(uint32_t, nc);       // N*C
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AveragePooling1dCustom, AveragePooling1dCustomTilingData)
} // namespace optiling
