/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file circular_pad_common.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_CIRCULAR_PAD_COMMON_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_CIRCULAR_PAD_COMMON_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_base.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CircularPadCommonTilingData)
  TILING_DATA_FIELD_DEF(int64_t, inputH);
  TILING_DATA_FIELD_DEF(int64_t, inputW);
  TILING_DATA_FIELD_DEF(int64_t, outputH);
  TILING_DATA_FIELD_DEF(int64_t, outputW);
  TILING_DATA_FIELD_DEF(int64_t, left);
  TILING_DATA_FIELD_DEF(int64_t, right);
  TILING_DATA_FIELD_DEF(int64_t, top);
  TILING_DATA_FIELD_DEF(int64_t, bottom);

  TILING_DATA_FIELD_DEF(int64_t, front);
  TILING_DATA_FIELD_DEF(int64_t, back);
  TILING_DATA_FIELD_DEF(int64_t, inputL);
  TILING_DATA_FIELD_DEF(int64_t, outputL);

  TILING_DATA_FIELD_DEF(int64_t, perCoreTaskNum);
  TILING_DATA_FIELD_DEF(int64_t, tailTaskNum);
  TILING_DATA_FIELD_DEF(int64_t, workspaceLen);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CircularPad, CircularPadCommonTilingData)
REGISTER_TILING_DATA_CLASS(CircularPadGrad, CircularPadCommonTilingData)

struct Tiling4CircularPadCommonCompileInfo {
  uint32_t coreNum;
  uint64_t ubSize;
  uint32_t sysWorkspaceSize;
};

class CircularPadCommonTiling{
public:
    explicit  CircularPadCommonTiling(gert::TilingContext* context) : context_(context) {};
    virtual ~CircularPadCommonTiling() = default;
    ge::graphStatus GetShapeAttrsInfo();
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetWorkspaceSize();
    ge::graphStatus PostTiling();
    uint64_t GetTilingKey();

    void DumpTilingInfo();
    void CalculateParams();
    void DivCore();
    void SetTilingKey();
    void SetTilingData();

    virtual ge::graphStatus CheckLeftAndRight() = 0;
    virtual ge::graphStatus CheckTopAndBottom() = 0;
    virtual ge::graphStatus CheckFrontAndBack() = 0;
    virtual ge::graphStatus CheckInput() = 0;
    virtual ge::graphStatus CheckDtype() = 0;

protected:
    int32_t shapeType = 0;
    int32_t dataType = 0;
    int32_t tSize = 0;
    bool dPad = false;

    int64_t inputH = 0;
    int64_t inputW = 0;
    int64_t inputL = 0;
    int64_t outputH = 0;
    int64_t outputW = 0;
    int64_t outputL = 0;
    int64_t front = 0;
    int64_t back = 0;
    int64_t left = 0;
    int64_t right = 0;
    int64_t top = 0;
    int64_t bottom = 0;
    int64_t perCoreTaskNum = 0;
    int64_t tailTaskNum = 0;
    int64_t workspaceLen = 0;

    int64_t inputLen = 0;
    int64_t outputWAlign = 0;
    int64_t inputWAlign = 0;
    int64_t outputLen = 0;
    int64_t inOutputH = 0;
    int64_t inOutputW = 0;
    int64_t inOutputL = 0;
    int64_t inOutputWAlign = 0;
    int64_t leftAlign = 0;
    int64_t rightAlign = 0;
    int64_t pLeft = 0;
    int64_t pRight = 0;
    int64_t pTop = 0;
    int64_t pBottom = 0;
    int64_t pFront = 0;
    int64_t pBack = 0;
    int64_t nLeft = 0;
    int64_t nRight = 0;
    int64_t nTop = 0;
    int64_t nBottom = 0;
    int64_t nFront = 0;
    int64_t nBack = 0;
    int64_t totalTasks = 1;
    uint32_t useCoreNum = 0;

    uint32_t coreNum = 0;
    uint64_t ubSize = 0;
    uint64_t sysWorkspaceSize = 0;
    uint64_t tilingKey_{0};

    gert::TilingContext *context_ = nullptr;
    CircularPadCommonTilingData tilingData_;
};
}
#endif
