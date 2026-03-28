/**
 * Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tiling_base.h
 * \brief
 */

#pragma once
#include <numeric>
#include <sstream>
#include <exe_graph/runtime/tiling_context.h>
#include <graph/utils/type_utils.h>
#include <tiling/platform/platform_ascendc.h>

#include "register/op_def_registry.h"
#include "flash_attention_score_with_large_head_dim_tiling.h"
#include "tiling_type.h"
namespace optiling {
namespace FA {
const int64_t FRACTAL_NUM = 16L;
const int64_t HIGH_PERF_API_BUFFER_MULTIPLE = 2L;
constexpr size_t WORK_SPACE_RESERVE_SIZE = 16 * 1024 * 1024;
const int64_t D_SPECIFIC_SIZE = 64L;
const int64_t BMM1_BASICBLOCK_N_128 = 128L;
const int64_t BMM1_BASICBLOCK_K_64 = 64L;
const int64_t BMM1_BASICBLOCK_K_128 = 128L;
const int64_t S2_NZTOND_SIZE_64 = 64L;
const int64_t SPACE_NUM_2 = 2L;
const int64_t SPACE_NUM_3 = 3L;
const int64_t SPACE_NUM_4 = 4L;
const int64_t HEAD_DIM_MAX_VALUE = 576L;

class FlashAttentionScoreWithLargeHeadDimTiling {
public:
    explicit FlashAttentionScoreWithLargeHeadDimTiling(gert::TilingContext *context) : context_(context)
    {
        context_ = context;
        tilingData.SetDataPtr(context_->GetRawTilingData()->GetData());
    }
    ~FlashAttentionScoreWithLargeHeadDimTiling() = default;

    ge::graphStatus DoTiling()
    {
        auto ret = GetShapeAttrsInfo();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        ret = GetPlatformInfo();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        if (!IsCapable()) {
            return ge::GRAPH_PARAM_INVALID;
        }
        ret = DoOpTiling();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        ret = DoLibApiTiling();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        ret = GetWorkspaceSize();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        ret = PostTiling();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        context_->SetTilingKey(0);
        return ge::GRAPH_SUCCESS;
    }

protected:
    bool IsCapable()
    {
        if (s2Size > s2sizeLimitMin) {
            return true;
        }
        LOG_PRINT("[error]s2Size need to be greater than %d, current s2Size is : %d\n", s2sizeLimitMin, s2Size);
        return false;
    }
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo();
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo();
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling();
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling();
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize();
    // 7、保存Tiling数据
    ge::graphStatus PostTiling();
    ge::graphStatus CheckContext();
    bool MatchTemplate();
    bool IsBasicBlockInSoftMax(const ge::Shape &shape) const;
    bool SetBmm1TilingInput(int64_t s1BasicBlock, int64_t s2BasicBlock,
                             matmul_tiling::MatmulApiTiling &bmm1);
    bool SetBmm2TilingInput(int64_t s1BasicBlock, int64_t s2BasicBlock, int64_t dBasicBlock,
                             matmul_tiling::MatmulApiTiling &bmm2);
    bool SetMatMulTiling(int64_t s1BasicBlock, int64_t s2BasicBlock, int64_t dBasicBlock,
                         matmul_tiling::MatmulApiTiling &bmm1, matmul_tiling::MatmulApiTiling &bmm2);
    bool SetMatMulTiling(int64_t s1BasicBlock, int64_t s2BasicBlock, int64_t dBasicBlock);
    void SetCoreParams();
    void SetMultiCoreParams();
    void SetSoftMaxTiling();
    
protected:
    gert::TilingContext *context_ = nullptr;
    AiCoreParams aicoreParams_{0, 0, 0, 0, 0, 0, 0};
    uint32_t aivNum;
    uint32_t aicNum;
    int64_t apiMaxUBSize = 0;
    int64_t actualUsedAivNum;
    int64_t calcTypeSize = ge::GetSizeByDataType(ge::DT_FLOAT);
    int64_t bSize = 0LL;
    int64_t gSize = 0LL;
    int64_t dSize = 0LL;
    int64_t n1Size = 0LL;
    int64_t n2Size = 0LL;
    int64_t s1Size = 0LL;
    int64_t s2Size = 0LL;
    int64_t s1StrideSize = 0LL; // query Shape S inner axes, for bmm1
    int64_t s2StrideSize = 0LL; // key Shape S inner axes, for bmm1
    float scaleValue = 1.0f;

    int64_t alignedS1 = 0LL;
    int64_t alignedS2 = 0LL;
    int64_t alignedD = 0LL;

    int64_t s1BasicBlock = std::numeric_limits<int64_t>::max();
    int64_t s2BasicBlock = std::numeric_limits<int64_t>::max();
    int64_t dBasicBlock = std::numeric_limits<int64_t>::max();
    int64_t nRatio = 8L;

    const char *templateName = "FlashAttentionScoreWithLargeHeadDimS1s2Bn2gs1";
    int64_t s2sizeLimitMin = 1024;
    FlashAttentionScoreWithLargeHeadDimTilingData tilingData;
};

ge::graphStatus FlashAttentionScoreWithLargeHeadDimTiling::CheckContext()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    CHECK_NULL(workspaces, return ge::GRAPH_FAILED);
    auto queryShape = context_->GetInputShape(0);
    auto queryDesc = context_->GetInputDesc(0);
    auto keyShape = context_->GetInputShape(1);
    auto keyDesc = context_->GetInputDesc(1);
    auto valueShape = context_->GetInputShape(2);
    auto valueDesc = context_->GetInputDesc(2);
    CHECK_NULL(queryShape, return ge::GRAPH_FAILED);
    CHECK_NULL(queryDesc, return ge::GRAPH_FAILED);
    CHECK_NULL(keyShape, return ge::GRAPH_FAILED);
    CHECK_NULL(keyDesc, return ge::GRAPH_FAILED);
    CHECK_NULL(valueShape, return ge::GRAPH_FAILED);
    CHECK_NULL(valueDesc, return ge::GRAPH_FAILED);
    CHECK_NULL(context_->GetRawTilingData(), return ge::GRAPH_FAILED);
    CHECK_NULL(context_->GetRawTilingData()->GetData(), return ge::GRAPH_FAILED);
    CHECK_RET(context_->GetRawTilingData()->GetCapacity() < tilingData.GetDataSize(),
                LOG_PRINT("context tiling data capacity %zu < actual tiling data size %zu.\n",
                context_->GetRawTilingData()->GetCapacity(), tilingData.GetDataSize());
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreWithLargeHeadDimTiling::GetShapeAttrsInfo()
{
    CHECK_RET(CheckContext() != ge::GRAPH_SUCCESS, LOG_PRINT("invalid context.");
               return ge::GRAPH_FAILED);

    // 获取属性值
    auto attrs = context_->GetAttrs();
    CHECK_NULL(attrs, return false);
    size_t idx = 0;
    auto scaleValuePtr = attrs->GetAttrPointer<float>(idx++);
    auto n1SizePtr = attrs->GetAttrPointer<uint32_t>(idx++);
    scaleValue = *scaleValuePtr;
    n1Size = *n1SizePtr;
    CHECK_RET(n1Size == 0, LOG_PRINT("Head num is zero."); return false);
    LOG_PRINT("attrs: scale_value[%f] head_num[%ld].\n", scaleValue, n1Size);

    // 根据属性值n1Size解析输入shape值
    auto &queryShape = context_->GetInputShape(0)->GetStorageShape();
    auto &keyShape = context_->GetInputShape(1)->GetStorageShape();
    bSize = queryShape.GetDim(0);
    s1Size = queryShape.GetDim(1);
    s2Size = keyShape.GetDim(1);
    int64_t h1 = queryShape.GetDim(2); // 2: H idx
    int64_t h2 = keyShape.GetDim(2);   // 2: H idx
    s1StrideSize = h1;
    s2StrideSize = h2;
    CHECK_RET(h1 == 0 || h2 == 0, LOG_PRINT("H is zero."); return false);
    CHECK_RET(h1 % n1Size != 0,
              LOG_PRINT("h1 [%ld] should be a multiple of n1Size [%ld].\n", h1, n1Size); return false);
    dSize = h1 / n1Size;
    gSize = h1 / h2;
    n2Size = h2 / dSize;
    CHECK_RET(gSize == 0, LOG_PRINT("gSize is zero."); return false);
    CHECK_RET(n2Size == 0, LOG_PRINT("n2Size is zero."); return false);
    CHECK_RET(dSize > HEAD_DIM_MAX_VALUE || dSize <= 0L,
               LOG_PRINT("dSize is not in range:(0, 512]."); return false);
    CHECK_RET(n1Size % n2Size != 0,
               LOG_PRINT("n1Size [%ld] should be a multiple of n2Size [%ld].\n", n1Size, n2Size);
               return false);

    alignedS1 = AlignUp(s1Size, FRACTAL_NUM);
    alignedS2 = AlignUp(s2Size, FRACTAL_NUM);
    alignedD = AlignUp(dSize, FRACTAL_NUM);
    CHECK_RET(alignedS1 <= 0, LOG_PRINT("invalid alignedS1 %ld.\n", alignedS1);
        return ge::GRAPH_FAILED);
    CHECK_RET(alignedS2 <= 0, LOG_PRINT("invalid alignedS2 %ld.\n", alignedS2);
        return ge::GRAPH_FAILED);
    CHECK_RET(alignedD <= 0, LOG_PRINT("invalid alignedD %ld.\n", alignedD);
        return ge::GRAPH_FAILED);
    tilingData.inputParams.set_bSize(bSize);
    tilingData.inputParams.set_n2Size(n2Size);
    tilingData.inputParams.set_gSize(gSize);
    tilingData.inputParams.set_s1Size(s1Size);
    tilingData.inputParams.set_s2Size(s2Size);
    tilingData.inputParams.set_dSize(dSize);
    tilingData.inputParams.set_scaleValue(scaleValue);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreWithLargeHeadDimTiling::GetPlatformInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    aivNum = ascendcPlatform.GetCoreNumAiv();
    aicNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, aicoreParams_.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, aicoreParams_.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, aicoreParams_.l0cSize);
    LOG_PRINT("get platform from compileInfo. aivNum(%u) aicNum(%u) ubSize(%lu) l1Size(%lu) l0cSize(%lu).\n",
              aivNum, aicNum, aicoreParams_.ubSize, aicoreParams_.l1Size, aicoreParams_.l0cSize);
    return ge::GRAPH_SUCCESS;
}

bool FlashAttentionScoreWithLargeHeadDimTiling::MatchTemplate()
{
    // s1.i: 默认64，当按64切分时，如果核外BNGS1out超vector核数时，S1.i设置为128
    // s2.i: 1024
    // UB Size calc logic: s1s2 * X * sizeof(T) + s1d * Y * sizeof(T) + s1 * expNum * 32 + s1 * 64 + apiTmp
    s1BasicBlock = std::min(64L, alignedS1);
    if (bSize * n1Size * gSize * CeilDiv(s1Size, s1BasicBlock) > aivNum) {
        s1BasicBlock = std::min(128L, alignedS1);
    }
    s2BasicBlock = std::min(128L, alignedS2);
    dBasicBlock = std::min(128L, alignedD);
    apiMaxUBSize = HIGH_PERF_API_BUFFER_MULTIPLE * s1BasicBlock * s2BasicBlock * sizeof(float);
    LOG_PRINT("[%s]final basic block: [%ld, %ld, %ld].\n", templateName, s1BasicBlock,
                s2BasicBlock, dBasicBlock);
    return true;
}

void FlashAttentionScoreWithLargeHeadDimTiling::SetCoreParams()
{
    // 矩阵size
    tilingData.coreParams.set_s1BaseSize(s1BasicBlock);
    tilingData.coreParams.set_s1OuterSize(CeilDivision(s1Size, s1BasicBlock));
    tilingData.coreParams.set_s2BaseSize(s2BasicBlock);
    tilingData.coreParams.set_nRatio(nRatio);
}

void FlashAttentionScoreWithLargeHeadDimTiling::SetMultiCoreParams()
{
    auto &multiCoreParams = tilingData.multiCoreParams;
    int64_t totalSize = bSize * n2Size * gSize * tilingData.coreParams.get_s1OuterSize();
    actualUsedAivNum = std::min(totalSize, static_cast<int64_t>(aivNum));
    multiCoreParams.set_totalSize(totalSize);
    multiCoreParams.set_splitFactorSize(CeilDivision(totalSize, actualUsedAivNum));
}

ge::graphStatus FlashAttentionScoreWithLargeHeadDimTiling::DoOpTiling()
{
    auto &inputParams = tilingData.inputParams;
    // 计算基本块大小
    MatchTemplate();
    // 根据基本块大小设置单核数据
    SetCoreParams();
    // 计算多核切分相关数据
    SetMultiCoreParams();
    return ge::GRAPH_SUCCESS;
}

bool FlashAttentionScoreWithLargeHeadDimTiling::SetBmm1TilingInput(int64_t s1BasicBlock, int64_t s2BasicBlock,
                                                       matmul_tiling::MatmulApiTiling &bmm1)
{
    bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, false);
    // B矩阵转置
    bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, true);
    bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    // 设置Matmul计算时的单次计算的形状singleM、singleN、singleK，单位为元素个数。
    bmm1.SetShape(std::min(s1BasicBlock, s1Size),
                    std::min(s2BasicBlock * tilingData.coreParams.get_nRatio(), s2Size), dSize);
    // 设置Matmul计算时的原始完整的形状M、N、K，单位为元素个数。
    bmm1.SetOrgShape(s1Size, s2BasicBlock * tilingData.coreParams.get_nRatio(), s1StrideSize, s2StrideSize);
    bmm1.SetBias(false);
    if (bmm1.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0) {
        return false;
    }
    if (dSize > BMM1_BASICBLOCK_K_64 && dSize <= BMM1_BASICBLOCK_K_128) {
        int64_t baseM = std::min(s1BasicBlock, AlignUp(s1Size, FRACTAL_NUM));
        bmm1.SetFixSplit(baseM, BMM1_BASICBLOCK_N_128, dSize);
    }
    return true;
}

bool FlashAttentionScoreWithLargeHeadDimTiling::SetBmm2TilingInput(int64_t s1BasicBlock, int64_t s2BasicBlock, int64_t dBasicBlock,
                        matmul_tiling::MatmulApiTiling &bmm2)
{
    int64_t singleM = std::min(s1BasicBlock, s1Size);
    bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, false);
    bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16, false);
    bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    bmm2.SetShape(singleM, dSize,
                  std::min(s2BasicBlock * tilingData.coreParams.get_nRatio(), s2Size));
    bmm2.SetOrgShape(s1Size, s2StrideSize, std::min(s2BasicBlock * tilingData.coreParams.get_nRatio(), s2Size),
                     s2StrideSize);
    bmm2.SetBias(false);
    if (bmm2.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0) {
        return false;
    }
    return true;
}

bool FlashAttentionScoreWithLargeHeadDimTiling::SetMatMulTiling(int64_t s1BasicBlock, int64_t s2BasicBlock,
                                                    int64_t dBasicBlock,
                                                    matmul_tiling::MatmulApiTiling &bmm1,
                                                    matmul_tiling::MatmulApiTiling &bmm2)
{
    if (!SetBmm1TilingInput(s1BasicBlock, s2BasicBlock, bmm1) ||
        !SetBmm2TilingInput(s1BasicBlock, s2BasicBlock, dBasicBlock, bmm2)) {
        return false;
    }

    if (bmm1.GetTiling(tilingData.bmm1TilingData) == -1) {
        LOG_PRINT("BMM1 tiling failed.");
        return false;
    }
    tilingData.bmm1TilingData.set_shareMode(0);
    tilingData.bmm1TilingData.set_shareL1Size(aicoreParams_.l1Size);
    tilingData.bmm1TilingData.set_shareL0CSize(aicoreParams_.l0cSize);

    if (bmm2.GetTiling(tilingData.bmm2TilingData) == -1) {
        LOG_PRINT("BMM2 tiling failed.");
        return false;
    }

    tilingData.bmm2TilingData.set_shareMode(0);
    tilingData.bmm2TilingData.set_shareL1Size(aicoreParams_.l1Size);
    tilingData.bmm2TilingData.set_shareL0CSize(aicoreParams_.l0cSize);

    return true;
}

bool FlashAttentionScoreWithLargeHeadDimTiling::SetMatMulTiling(int64_t s1BasicBlock, int64_t s2BasicBlock,
                                                    int64_t dBasicBlock)
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        matmul_tiling::MatmulApiTiling bmm1(ascendcPlatform);
        matmul_tiling::MatmulApiTiling bmm2(ascendcPlatform);
        return SetMatMulTiling(s1BasicBlock, s2BasicBlock, dBasicBlock, bmm1, bmm2);
    } else {
        LOG_PRINT("platform info is null, use default info to generate matmul tiling.");
        matmul_tiling::MatmulApiTiling bmm1;
        matmul_tiling::MatmulApiTiling bmm2;
        return SetMatMulTiling(s1BasicBlock, s2BasicBlock, dBasicBlock, bmm1, bmm2);
    }
}

bool FlashAttentionScoreWithLargeHeadDimTiling::IsBasicBlockInSoftMax(const ge::Shape &shape) const
{
    int64_t lastAxis = shape.GetDim(shape.GetDimNum() - 1);
    // last axis should be less than 2048 and fullfil 64 times
    int64_t basicLastAxis = 64;
    int64_t lastAxisNum = 2048;
    if (lastAxis > lastAxisNum || lastAxis % basicLastAxis != 0) {
        return false;
    }
    int64_t preAxes = 1;
    for (size_t idx = 0; idx < shape.GetDimNum() - 1; ++idx) {
        preAxes *= shape.GetDim(idx);
    }
    // all axes except last one should be 8 times
    return preAxes % 8 == 0;
}

void FlashAttentionScoreWithLargeHeadDimTiling::SetSoftMaxTiling()
{
    auto softmaxShape = ge::Shape({s1BasicBlock / nRatio, s2BasicBlock * nRatio});

    AscendC::SoftMaxFlashV2TilingFunc(softmaxShape, calcTypeSize, sizeof(float), apiMaxUBSize,
                                        tilingData.softmaxFlashTilingData, true, IsBasicBlockInSoftMax(softmaxShape));
}

ge::graphStatus FlashAttentionScoreWithLargeHeadDimTiling::DoLibApiTiling()
{
    if (!SetMatMulTiling(s1BasicBlock, s2BasicBlock, dBasicBlock)) {
        return ge::GRAPH_FAILED;
    }
    SetSoftMaxTiling();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreWithLargeHeadDimTiling::GetWorkspaceSize()
{
    auto &coreParams = tilingData.coreParams;
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    int64_t bmm1Bytes = coreParams.get_nRatio() * s1BasicBlock * s2BasicBlock * calcTypeSize;

    // dSize小于64的场景，无需切D， workspace占用较小
    if (dSize <= D_SPECIFIC_SIZE) {
        // stage1占用2倍的空间，stage2占用2倍空间
        workspaces[0] = static_cast<size_t>((bmm1Bytes * SPACE_NUM_2 +
                        SPACE_NUM_2 * coreParams.get_s1BaseSize() * alignedD * calcTypeSize) * aivNum) +
                        WORK_SPACE_RESERVE_SIZE;
        if (s2Size % S2_NZTOND_SIZE_64 != 0) {
            workspaces[0] = static_cast<size_t>((bmm1Bytes * SPACE_NUM_3 +
                            SPACE_NUM_2 * coreParams.get_s1BaseSize() * alignedD * calcTypeSize) * aivNum) +
                            WORK_SPACE_RESERVE_SIZE;
        }
    } else {
        // 切D场景，stage1占用2倍的空间，stage2占用4倍空间
        workspaces[0] = static_cast<size_t>((bmm1Bytes * SPACE_NUM_2 +
                        SPACE_NUM_4 * coreParams.get_s1BaseSize() * alignedD * calcTypeSize) * aivNum) +
                        WORK_SPACE_RESERVE_SIZE;
        if (s2Size % S2_NZTOND_SIZE_64 != 0) {
            workspaces[0] = static_cast<size_t>((bmm1Bytes * SPACE_NUM_3 +
                            SPACE_NUM_4 * coreParams.get_s1BaseSize() * alignedD * calcTypeSize) * aivNum) +
                            WORK_SPACE_RESERVE_SIZE;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreWithLargeHeadDimTiling::PostTiling()
{
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize()); // already check capcity in CheckContext
    auto blockDim = optiling::CalcTschBlockDim(actualUsedAivNum, aicNum, aivNum);
    context_->SetBlockDim(blockDim);
    auto &inputParams = tilingData.inputParams;
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    LOG_PRINT("[%s] tiling data size: %zu", templateName, tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

} // namespace FA
} // namespace optiling
