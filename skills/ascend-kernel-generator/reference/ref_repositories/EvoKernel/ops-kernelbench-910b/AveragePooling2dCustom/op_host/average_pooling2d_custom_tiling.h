
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AveragePooling2dCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, c);
    TILING_DATA_FIELD_DEF(uint32_t, h_in);
    TILING_DATA_FIELD_DEF(uint32_t, w_in);
    TILING_DATA_FIELD_DEF(uint32_t, h_out);
    TILING_DATA_FIELD_DEF(uint32_t, w_out);

    TILING_DATA_FIELD_DEF(uint32_t, totalY);         // N*C*Ho*Wo (fits in u32 for this benchmark)
    TILING_DATA_FIELD_DEF(uint32_t, elemsPerBlock);  // linear output elements handled per block (tail-safe)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AveragePooling2dCustom, AveragePooling2dCustomTilingData)
} // namespace optiling
