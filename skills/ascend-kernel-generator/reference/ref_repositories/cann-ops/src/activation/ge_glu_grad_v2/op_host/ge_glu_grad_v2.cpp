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
 * \file ge_glu_grad_v2.cpp
 * \brief tiling
 */
#include <map>
#include <vector>
#include <string>
#include "ge_glu_grad_v2_tiling.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

namespace ops {
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    std::printf("nullptr error!");                                                               \
    return ge::GRAPH_FAILED;                                                                     \
  }
}  // namespace ops

namespace optiling {

#define unlikely(x) __builtin_expect((x), 0)
#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)
#define OP_LOGE_IF(condition, return_value, op_name, fmt, ...)                                                 \
  static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool"); \
  do {                                                                                                         \
    if (unlikely(condition)) {                                                                                 \
      OP_LOGE(op_name, fmt, ##__VA_ARGS__);                                                                    \
      return return_value;                                                                                     \
    }                                                                                                          \
  } while (0)
#define OP_TILING_CHECK(cond, log_func, expr)  \
  do {                                         \
    if (cond) {                                \
    std::printf(log_func);                     \
    expr;                                      \
    }                                          \
  } while (0)

template <typename T>
inline T* GetCompileInfoPtr(gert::TilingParseContext* context) {
  return context->GetCompiledInfo<T>();
}

constexpr char NODE_NAME[] = "GeGluGradV2";

constexpr uint32_t DY_INDEX = 0;
constexpr uint32_t X_INDEX = 1;
constexpr uint32_t GELU_INDEX = 2;
constexpr uint32_t DX_INDEX = 0;

constexpr uint32_t DIM_ATTR_INDEX = 0;
constexpr uint32_t APPROXIMATE_ATTR_INDEX = 1;
constexpr uint32_t ACTIVATE_LEFT_ATTR_INDEX = 2;

constexpr uint32_t BATCH_MODE = 1;

/* Tanh */
constexpr int32_t TANH_BUF_CNT_FP16 = 5 * 2 + 6;
constexpr int32_t TANH_BUF_CNT_BFP16 = 7 * 2 + 4;
constexpr int32_t TANH_BUF_CNT_FP32 = 11;

/* Erf */
constexpr int32_t ERF_BUF_CNT_FP16 = 5 * 2 + 6;
constexpr int32_t ERF_BUF_CNT_BFP16 = 7 * 2 + 4;
constexpr int32_t ERF_BUF_CNT_FP32 = 11;

constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t TRANSPOSE_REPEAT_SIZE = 512;
constexpr int32_t WORK_SPACE_SIZE = 16 * 1024 * 1024;
constexpr int32_t NUM_ONE = 1;
constexpr int32_t NUM_TWO = 2;
constexpr int32_t NUM_HUNDRED = 100;

static const map<ge::DataType, int32_t> DTYPE_BUF_CNT_MAP_TANH = {
    {ge::DT_BF16, TANH_BUF_CNT_BFP16}, {ge::DT_FLOAT16, TANH_BUF_CNT_FP16}, {ge::DT_FLOAT, TANH_BUF_CNT_FP32}};
static const map<ge::DataType, int32_t> DTYPE_BUF_CNT_MAP_ERF = {
    {ge::DT_BF16, ERF_BUF_CNT_BFP16}, {ge::DT_FLOAT16, ERF_BUF_CNT_FP16}, {ge::DT_FLOAT, ERF_BUF_CNT_FP32}};

class GeGluGradV2Tiling {
public:
    explicit GeGluGradV2Tiling(gert::TilingContext* context) : tilingContext(context){};
    ge::graphStatus RunTiling4GeGluGradV2(gert::TilingContext* context);

private:
    ge::graphStatus Init(gert::TilingContext* context);
    ge::graphStatus CheckParams(gert::TilingContext* context);
    void FillTilingData();

    template <typename T1, typename T2>
    inline T1 CeilDiv(T1 a, T2 b) const {
        a = int64_t(a);
        b = int64_t(b);
        return T1(b == 0 ? a : (a + b - 1) / b);
    };

    template <typename T1, typename T2>
    inline T1 AlignA2B(T1 a, T2 b) const {
        a = int64_t(a);
        b = int64_t(b);
        return T1(b == 0 ? a : (a / b) * b);
    };

    void CalcValueNM();
    ge::graphStatus CaclMaxProcessCount();
    void ProcessTilingCore(gert::TilingContext* context);

private:
    GeGluGradV2TilingData tilingData;
    GeGluGradV2TilingKey tilingKey;
    gert::TilingContext* tilingContext = nullptr;
    const GeGluGradV2CompileInfo* ptrCompileInfo = nullptr;

    // input output infos
    gert::Shape dyShape;
    gert::Shape xShape;
    gert::Shape geluShape;
    gert::Shape dxShape;
    int64_t dimAttr = -1;
    int64_t approximateAttr = 1;
    bool activateLeftAttr = false;
    platform_ascendc::SocVersion curShortSocName_;
    uint64_t ubSizePlatForm;
    
    /**
     * The meanings of valueN and valueM are as follows:
     * Shape(A, B, C) of input x, dim=1 ==> valueN=A, valueM=B*C//2
     * Shape(A, B, C) of input x, dim=-1 ==> valueN=A*B, valueM=C//2
     * Shape(A, B, C, D) of input x, dim=2 ==> valueN=A*B, valueM=C*D//2
     */
    int64_t valueN = 1;
    int64_t valueM = 1;
    ge::DataType dyDtype = ge::DT_UNDEFINED;
    int32_t dtypeSize = 0;

    // tiling params
    int64_t maxProcCount = 0;
    int32_t needCoreNum = 0;
    int64_t loopNumPerCore = 0;
    int64_t tailCoreIndex = 0;
    int64_t tailUbLoopNum = 0;
    int64_t groupNum = 0;
};

ge::graphStatus GeGluGradV2Tiling::RunTiling4GeGluGradV2(gert::TilingContext* context) {
    if (Init(context) != ge::GRAPH_SUCCESS) {
        OP_LOGE(NODE_NAME, "Init failed");
        return ge::GRAPH_FAILED;
    }
    if (CheckParams(context) != ge::GRAPH_SUCCESS) {
        OP_LOGE(NODE_NAME, "CheckParams failed");
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    curShortSocName_ = ascendcPlatform.GetSocVersion();

    CalcValueNM();
    
    OP_LOGD(NODE_NAME, "Platform info, ubSizePlatForm:%lu, totalCoreNum:%d, curSocVersion:%d.",
            ubSizePlatForm, ascendcPlatform.GetCoreNumAiv(), static_cast<int32_t>(curShortSocName_));
    if (CaclMaxProcessCount() != ge::GRAPH_SUCCESS) {
        OP_LOGE(NODE_NAME, "CaclMaxProcessCount failed");
        return ge::GRAPH_FAILED;
    }

    ProcessTilingCore(context);

    tilingContext->SetBlockDim(needCoreNum);
    tilingContext->SetTilingKey(static_cast<uint64_t>(tilingKey));
    FillTilingData();
    size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
    workspaces[0] = WORK_SPACE_SIZE + ascendcPlatform.GetCoreNumAiv() * BLOCK_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeGluGradV2Tiling::Init(gert::TilingContext* context) {
    auto inputDy = tilingContext->GetInputTensor(DY_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDy);
    dyShape = inputDy->GetStorageShape();
    dyDtype = tilingContext->GetInputDesc(DY_INDEX)->GetDataType();

    auto inputX = tilingContext->GetInputTensor(X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(tilingContext, inputX);
    xShape = inputX->GetStorageShape();

    auto inputYgelu = tilingContext->GetInputTensor(GELU_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(tilingContext, inputYgelu);
    geluShape = inputYgelu->GetStorageShape();

    auto outputDx = tilingContext->GetOutputShape(DX_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDx);
    dxShape = outputDx->GetStorageShape();

    auto attrs = tilingContext->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(tilingContext, attrs);
    const int64_t* ptrDim = attrs->GetAttrPointer<int64_t>(DIM_ATTR_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(tilingContext, ptrDim);
    dimAttr = *ptrDim;
    const int64_t* ptrApproximate = attrs->GetAttrPointer<int64_t>(APPROXIMATE_ATTR_INDEX);

    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    const bool is310p = curShortSocName_ == platform_ascendc::SocVersion::ASCEND310P;
    OPS_CHECK_NULL_WITH_CONTEXT(tilingContext, ptrApproximate);
    approximateAttr = *ptrApproximate;

    if (approximateAttr == 0 && is310p) {
        OP_LOGE(NODE_NAME,"approximate only support 1(Tanh) in 310P");
        return ge::GRAPH_FAILED;
    }

    const bool* ptrActivateLeft = attrs->GetAttrPointer<bool>(ACTIVATE_LEFT_ATTR_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(tilingContext, ptrActivateLeft);
    activateLeftAttr = *ptrActivateLeft;

    OP_LOGD(NODE_NAME, "Attr info: dimAttr: %ld, approximateAttr: %ld, activateLeftAttr: %s, dyDtype: %d.", dimAttr,
            approximateAttr, activateLeftAttr ? "true" : "false", static_cast<int32_t>(dyDtype));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeGluGradV2Tiling::CheckParams(gert::TilingContext* context) {
    if (dyDtype != ge::DT_BF16 && dyDtype != ge::DT_FLOAT16 && dyDtype != ge::DT_FLOAT) {
        OP_LOGE(NODE_NAME, "Data type support only float16, bfloat16, float32");
        return ge::GRAPH_FAILED;
    }

    // 310P donot support bfloat16 and erf mode
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    const bool is310p = curShortSocName_ == platform_ascendc::SocVersion::ASCEND310P;
    if (dyDtype == ge::DT_BF16 && is310p) {
        OP_LOGE(NODE_NAME, "Data type donot support only bfloat16 in 310P");
        return ge::GRAPH_FAILED;
    }

    dtypeSize = ge::GetSizeByDataType(dyDtype);
    if (dtypeSize <= 0) {
        OP_LOGE(NODE_NAME, "dtypeSize[%d] is invalid");
        return ge::GRAPH_FAILED;
    }

    auto xDtype = tilingContext->GetInputDesc(X_INDEX)->GetDataType();
    auto geluDtype = tilingContext->GetInputDesc(GELU_INDEX)->GetDataType();
    auto dxDtype = tilingContext->GetInputDesc(DX_INDEX)->GetDataType();
    if (dyDtype != geluDtype || xDtype != dxDtype || dyDtype != xDtype) {
        OP_LOGE(NODE_NAME,"The dtype of input should be same");
        return ge::GRAPH_FAILED;
    }

    size_t xDimNum = xShape.GetDimNum();
    dimAttr = dimAttr < 0 ? xDimNum + dimAttr : dimAttr;
    if (dimAttr < 0 || dimAttr >= static_cast<int64_t>(xDimNum)) {
        OP_LOGE(NODE_NAME, "Dim %ld is not in [0, %lu).", dimAttr, xDimNum);
        return ge::GRAPH_FAILED;
    }

    gert::Shape tempShape = dyShape;
    tempShape.SetDim(dimAttr, NUM_TWO * dyShape.GetDim(dimAttr));
    if (dyShape != geluShape || xShape != dxShape || tempShape != xShape) {
        OP_LOGE(NODE_NAME, "The input-output shape does not satisfy the operator constraint.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

void GeGluGradV2Tiling::FillTilingData() {
    tilingData.set_approximate(static_cast<int32_t>(approximateAttr));
    tilingData.set_activateLeft(static_cast<int32_t>(activateLeftAttr));
    tilingData.set_maxProcCount(maxProcCount);
    tilingData.set_valueN(valueN);
    tilingData.set_valueM(valueM);
    tilingData.set_needCoreNum(needCoreNum);
    tilingData.set_loopNumPerCore(loopNumPerCore);
    tilingData.set_tailCoreIndex(tailCoreIndex);
    tilingData.set_tailUbLoopNum(tailUbLoopNum);
    tilingData.set_groupNum(groupNum);

    tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(),
                            tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    OP_LOGD(NODE_NAME,
            "Tiling data is maxProcCount:%ld, valueN:%ld, valueM:%ld, needCoreNum:%ld, loopNumPerCore:%ld, "
            "tailCoreIndex:%ld, tailUbLoopNum:%ld, groupNum:%ld, tilingKey:%lu.",
            tilingData.get_maxProcCount(), tilingData.get_valueN(), tilingData.get_valueM(),
            tilingData.get_needCoreNum(), tilingData.get_loopNumPerCore(), tilingData.get_tailCoreIndex(),
            tilingData.get_tailUbLoopNum(), tilingData.get_groupNum(), static_cast<uint64_t>(tilingKey));
}

void GeGluGradV2Tiling::CalcValueNM() {
    for (int64_t i = 0; i < dimAttr; ++i) {
        valueN *= dyShape.GetDim(i);
    }
    for (int64_t i = dimAttr; i < int64_t(dyShape.GetDimNum()); ++i) {
        valueM *= dyShape.GetDim(i);
    }
}

ge::graphStatus GeGluGradV2Tiling::CaclMaxProcessCount() {
    if (approximateAttr == NUM_ONE) {
        const auto iter = DTYPE_BUF_CNT_MAP_TANH.find(dyDtype);
        maxProcCount = AlignA2B(ubSizePlatForm / iter->second, BLOCK_SIZE) / dtypeSize;
        tilingKey = GeGluGradV2TilingKey::TILING_KEY_TANH_101;
    } else {
        const auto iter = DTYPE_BUF_CNT_MAP_ERF.find(dyDtype);
        maxProcCount = AlignA2B(ubSizePlatForm / iter->second, BLOCK_SIZE) / dtypeSize;
        tilingKey = GeGluGradV2TilingKey::TILING_KEY_ERF_701;
    }

    if (dyDtype == ge::DT_FLOAT16) {
        tilingKey = static_cast<GeGluGradV2TilingKey>(static_cast<int32_t>(tilingKey) + NUM_HUNDRED);
    } else if (dyDtype == ge::DT_FLOAT) {
        tilingKey = static_cast<GeGluGradV2TilingKey>(static_cast<int32_t>(tilingKey) + NUM_TWO * NUM_HUNDRED);
    }

    return ge::GRAPH_SUCCESS;
}

void GeGluGradV2Tiling::ProcessTilingCore(gert::TilingContext* context) {
    int64_t ubLoopNum = 0;
    int64_t repeatDataCount = TRANSPOSE_REPEAT_SIZE / dtypeSize;
    int64_t maxPerfCount = maxProcCount / repeatDataCount;
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    if (curShortSocName_ == platform_ascendc::SocVersion::ASCEND910B && valueM <= maxPerfCount) {
        tilingKey = static_cast<GeGluGradV2TilingKey>(static_cast<int32_t>(tilingKey) + NUM_TWO);
        groupNum = AlignA2B(maxProcCount / valueM, repeatDataCount);
        ubLoopNum = CeilDiv(valueN, groupNum);
        tailUbLoopNum = valueN % groupNum;
    } else if (valueM <= maxProcCount) {
        int64_t alignValueM = CeilDiv(valueM, (BLOCK_SIZE / dtypeSize)) * (BLOCK_SIZE / dtypeSize);
        groupNum = maxProcCount / alignValueM;
        ubLoopNum = CeilDiv(valueN, groupNum);
        tailUbLoopNum = valueN % groupNum;
    } else {
        groupNum = CeilDiv(valueM, maxProcCount);
        ubLoopNum = valueN * groupNum;
        tilingKey = static_cast<GeGluGradV2TilingKey>(static_cast<int32_t>(tilingKey) + NUM_ONE);
    }

    needCoreNum = ubLoopNum < ascendcPlatform.GetCoreNumAiv() ? ubLoopNum : ascendcPlatform.GetCoreNumAiv();
    if (needCoreNum < ascendcPlatform.GetCoreNumAiv()) {
        loopNumPerCore = 0;
        tailCoreIndex = tailUbLoopNum != 0 ? needCoreNum - 1 : needCoreNum;
    } else {
        loopNumPerCore = ubLoopNum / ascendcPlatform.GetCoreNumAiv();
        int64_t modValue = ubLoopNum % ascendcPlatform.GetCoreNumAiv();
        if (modValue != 0) {
            tailCoreIndex = tailUbLoopNum != 0 ? modValue - 1 : modValue;
        } else {
            loopNumPerCore -= 1;
            tailCoreIndex = tailUbLoopNum != 0 ? ascendcPlatform.GetCoreNumAiv() - 1 : ascendcPlatform.GetCoreNumAiv();
        }
    }
}

ge::graphStatus Tiling4GeGluGradV2(gert::TilingContext* context) {
    OP_LOGD(NODE_NAME, "Tiling4GeGluGradV2 tiling begin.");
    context->SetScheduleMode(BATCH_MODE);
    GeGluGradV2Tiling tilingObject(context);
    if (tilingObject.RunTiling4GeGluGradV2(context) != ge::GRAPH_SUCCESS) {
        OP_LOGE(NODE_NAME, "RunTiling4GeGluGradV2 failed");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(NODE_NAME, "Tiling4GeGluGradV2 tiling end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4GeGluGradV2(gert::TilingParseContext* context) {
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    if (totalCoreNum <= 0) {
        OP_LOGE(NODE_NAME, "TilingPrepare4GeGluGradV2 get core num failed");
        return ge::GRAPH_FAILED;
    }
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    if (ubSizePlatForm <= 0) {
        OP_LOGE(NODE_NAME, "TilingPrepare4GeGluGradV2 get ub size failed");
        return ge::GRAPH_FAILED;
    }
    auto curSocVersion = ascendcPlatform.GetSocVersion();
    return ge::GRAPH_SUCCESS;
}


IMPL_OP_OPTILING(GeGluGradV2).Tiling(Tiling4GeGluGradV2).TilingParse<GeGluGradV2CompileInfo>(TilingPrepare4GeGluGradV2);

}  // namespace optiling
