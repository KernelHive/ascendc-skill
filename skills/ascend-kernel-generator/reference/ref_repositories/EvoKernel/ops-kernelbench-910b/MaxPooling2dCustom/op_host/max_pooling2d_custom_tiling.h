
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MaxPooling2dCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, c);
    TILING_DATA_FIELD_DEF(uint32_t, h_in);
    TILING_DATA_FIELD_DEF(uint32_t, w_in);
    TILING_DATA_FIELD_DEF(uint32_t, h_out);
    TILING_DATA_FIELD_DEF(uint32_t, w_out);

    TILING_DATA_FIELD_DEF(uint32_t, totalY);         // N*C*Ho*Wo
    TILING_DATA_FIELD_DEF(uint32_t, elemsPerBlock);  // ceil-div(totalY, blockDim)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MaxPooling2dCustom, MaxPooling2dCustomTilingData)
} // namespace optiling
