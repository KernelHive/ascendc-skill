
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(Conv3dGroupNormMinClampDropoutCustomTilingData)
  // runtime
  TILING_DATA_FIELD_DEF(uint32_t, N);

  // specialized constants for kernel
  TILING_DATA_FIELD_DEF(uint32_t, Cin);
  TILING_DATA_FIELD_DEF(uint32_t, Din);
  TILING_DATA_FIELD_DEF(uint32_t, Hin);
  TILING_DATA_FIELD_DEF(uint32_t, Win);

  TILING_DATA_FIELD_DEF(uint32_t, Cout);
  TILING_DATA_FIELD_DEF(uint32_t, K);

  TILING_DATA_FIELD_DEF(uint32_t, Dout);
  TILING_DATA_FIELD_DEF(uint32_t, Hout);
  TILING_DATA_FIELD_DEF(uint32_t, Wout);

  TILING_DATA_FIELD_DEF(uint32_t, G);
  TILING_DATA_FIELD_DEF(uint32_t, CperG);

  TILING_DATA_FIELD_DEF(uint32_t, DHW);
  TILING_DATA_FIELD_DEF(uint32_t, elemsPerG);
  TILING_DATA_FIELD_DEF(float, invElemsPerG);
  TILING_DATA_FIELD_DEF(float, eps);

  TILING_DATA_FIELD_DEF(float, minValue);
  TILING_DATA_FIELD_DEF(float, clampMin);
  TILING_DATA_FIELD_DEF(float, clampMax);

  // Dropout params
  TILING_DATA_FIELD_DEF(float, dropoutP);
  TILING_DATA_FIELD_DEF(uint32_t, dropThresholdU32);
  TILING_DATA_FIELD_DEF(float, invKeepProb);

  // parallel mapping
  TILING_DATA_FIELD_DEF(uint32_t, tasksPerSample); // == G
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv3dGroupNormMinClampDropoutCustom,
                           Conv3dGroupNormMinClampDropoutCustomTilingData)

} // namespace optiling
