
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(DeepNarrowMlpCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalX);
    TILING_DATA_FIELD_DEF(uint32_t, totalW);
    TILING_DATA_FIELD_DEF(uint32_t, totalB);
    TILING_DATA_FIELD_DEF(uint32_t, totalY);

    TILING_DATA_FIELD_DEF(uint32_t, batch);
    TILING_DATA_FIELD_DEF(uint32_t, inSize);
    TILING_DATA_FIELD_DEF(uint32_t, hiddenSize);
    TILING_DATA_FIELD_DEF(uint32_t, outSize);
    TILING_DATA_FIELD_DEF(uint32_t, numHidden);

    // Row-parallel scheduling
    TILING_DATA_FIELD_DEF(uint32_t, rowsPerBlock);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DeepNarrowMlpCustom, DeepNarrowMlpCustomTilingData)
} // namespace optiling
