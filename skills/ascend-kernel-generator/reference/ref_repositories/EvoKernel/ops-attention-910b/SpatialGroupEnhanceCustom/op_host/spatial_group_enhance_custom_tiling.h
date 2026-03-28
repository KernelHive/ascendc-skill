
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SpatialGroupEnhanceCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, B);
    TILING_DATA_FIELD_DEF(uint32_t, C);
    TILING_DATA_FIELD_DEF(uint32_t, H);
    TILING_DATA_FIELD_DEF(uint32_t, W);
    TILING_DATA_FIELD_DEF(uint32_t, G);
    TILING_DATA_FIELD_DEF(uint32_t, Cg);
    TILING_DATA_FIELD_DEF(uint32_t, HW);
    TILING_DATA_FIELD_DEF(uint32_t, groupsTotal); // B*G
    TILING_DATA_FIELD_DEF(uint32_t, hwAlign);
    TILING_DATA_FIELD_DEF(float, invHW);
    TILING_DATA_FIELD_DEF(float, invCg);
    TILING_DATA_FIELD_DEF(float, epsilon);
    TILING_DATA_FIELD_DEF(uint32_t, sigTmpBytes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SpatialGroupEnhanceCustom, SpatialGroupEnhanceCustomTilingData)
} // namespace optiling
