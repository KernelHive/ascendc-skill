
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LCTCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, N);
    TILING_DATA_FIELD_DEF(uint32_t, C);
    TILING_DATA_FIELD_DEF(uint32_t, H);
    TILING_DATA_FIELD_DEF(uint32_t, W);
    TILING_DATA_FIELD_DEF(uint32_t, HW);
    TILING_DATA_FIELD_DEF(uint32_t, CHW);
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF(uint32_t, CAlign);
    TILING_DATA_FIELD_DEF(float, invHW);
    TILING_DATA_FIELD_DEF(uint32_t, sigTmpBytes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LCTCustom, LCTCustomTilingData)
} // namespace optiling
