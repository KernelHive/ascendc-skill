
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(LstmHnCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalX);
    TILING_DATA_FIELD_DEF(uint32_t, totalH0);
    TILING_DATA_FIELD_DEF(uint32_t, totalC0);
    TILING_DATA_FIELD_DEF(uint32_t, totalWih);
    TILING_DATA_FIELD_DEF(uint32_t, totalWhh);
    TILING_DATA_FIELD_DEF(uint32_t, totalBih);
    TILING_DATA_FIELD_DEF(uint32_t, totalBhh);
    TILING_DATA_FIELD_DEF(uint32_t, totalHn);

    TILING_DATA_FIELD_DEF(uint32_t, B);
    TILING_DATA_FIELD_DEF(uint32_t, S);
    TILING_DATA_FIELD_DEF(uint32_t, I);
    TILING_DATA_FIELD_DEF(uint32_t, H);
    TILING_DATA_FIELD_DEF(uint32_t, L);

    TILING_DATA_FIELD_DEF(uint32_t, blockB);
    TILING_DATA_FIELD_DEF(uint32_t, totalB);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LstmHnCustom, LstmHnCustomTilingData)
} // namespace optiling
