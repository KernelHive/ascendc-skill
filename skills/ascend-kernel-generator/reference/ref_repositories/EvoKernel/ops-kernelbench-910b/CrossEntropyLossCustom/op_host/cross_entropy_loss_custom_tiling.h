
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CrossEntropyLossCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, N);
    TILING_DATA_FIELD_DEF(uint32_t, C);
    TILING_DATA_FIELD_DEF(uint32_t, tileC);
    TILING_DATA_FIELD_DEF(uint32_t, tilesPerRow);
    TILING_DATA_FIELD_DEF(float, invN);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CrossEntropyLossCustom, CrossEntropyLossCustomTilingData)
} // namespace optiling
