
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MhcProjectorCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, B);
  TILING_DATA_FIELD_DEF(uint32_t, T);
  TILING_DATA_FIELD_DEF(uint32_t, N);
  TILING_DATA_FIELD_DEF(uint32_t, C);
  TILING_DATA_FIELD_DEF(uint32_t, BT);     // B*T
  TILING_DATA_FIELD_DEF(uint32_t, F);      // N*C
  TILING_DATA_FIELD_DEF(uint32_t, NN);     // N*N
  TILING_DATA_FIELD_DEF(uint32_t, Fpad);   // ceil(F,8)
  TILING_DATA_FIELD_DEF(uint32_t, Npad);   // ceil(N,8)
  TILING_DATA_FIELD_DEF(uint32_t, NNpad);  // ceil(NN,8)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MhcProjectorCustom, MhcProjectorCustomTilingData)

} // namespace optiling
