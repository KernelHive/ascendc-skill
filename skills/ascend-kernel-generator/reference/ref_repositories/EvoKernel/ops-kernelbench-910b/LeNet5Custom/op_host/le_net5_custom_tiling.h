
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(LeNet5CustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batch);
    TILING_DATA_FIELD_DEF(uint32_t, num_classes);
    TILING_DATA_FIELD_DEF(uint32_t, block_dim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LeNet5Custom, LeNet5CustomTilingData)

} // namespace optiling
