
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MhcPostBlockCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, N);   // num_tokens
  TILING_DATA_FIELD_DEF(uint32_t, S);   // hc_mult
  TILING_DATA_FIELD_DEF(uint32_t, H);   // hidden_size
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MhcPostBlockCustom, MhcPostBlockCustomTilingData)

} // namespace optiling
