
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatmulWithTransposedACustomTilingData)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingData);
// how many N-tiles each block processes per iteration (amortize setup / reduce gaps)
TILING_DATA_FIELD_DEF(uint32_t, nGroup);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatmulWithTransposedACustom, MatmulWithTransposedACustomTilingData)
} // namespace optiling
