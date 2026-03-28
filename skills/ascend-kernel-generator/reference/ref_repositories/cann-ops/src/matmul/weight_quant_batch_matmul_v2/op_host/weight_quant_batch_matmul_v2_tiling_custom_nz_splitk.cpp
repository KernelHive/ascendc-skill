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
 * \file weight_quant_batch_matmul_v2_tiling_custom_nz_splitk.cpp
 * \brief
 */

#include "weight_quant_batch_matmul_v2_tiling_custom_nz_splitk.h"
#include "weight_quant_batch_matmul_v2_tiling_tool.h"
#include "weight_quant_batch_matmul_v2_tiling_key.h"
#include "weight_quant_batch_matmul_v2_white_list.h"

namespace optiling {

constexpr uint64_t M_MIN_LIMIT = 65UL;
constexpr uint64_t M_MAX_LIMIT = 256UL;
constexpr uint64_t MAX_SHAPE_DIM = 65535UL;
constexpr uint64_t SHAPE_ALIGNED_FACTOR = 64UL;
constexpr uint64_t SINGLE_K = 512UL; // 单次计算K方向数据量
constexpr uint64_t VECTOR_SINGLE_N = 32UL; // 单次计算N方向数据量
constexpr uint64_t CUBE_BASE_M = 128UL; // cube M方向基本块
constexpr uint64_t CUBE_BASE_N = 128UL; // cube N方向基本块
constexpr uint64_t CUBE_BASE_K = 128UL; // cube K方向基本块
constexpr uint64_t CACHE_NUM = 4UL;
constexpr uint64_t VEC_CUBE_RATIO = 2;
constexpr uint64_t DEFAULT_SINGLE_CORE_SIZE = 1024;
constexpr uint64_t L2_SIZE = 67108864;


const std::set<WhiteListShape> NETWORK_UNALIGN_WHITE_LIST = {
    // m和核数为1，表示该维度不参与匹配
    {1, 8192, 13750, false, false, true, 1},
    {1, 6144, 20708, false, false, true, 1},
    {1, 8192, 16032, false, false, true, 1},
    {1, 3696, 8192, false, false, true, 1},
    {1, 8192, 7392, true, false, true, 1}
};

void WeightQuantBatchMatmulV2CustomNzSplitK::Reset()
{
    WeightQuantBatchMatmulV2Tiling::Reset();
    cubeSingleN_ = 128UL;
    al1FullLoad_ = false;

    OP_TILING_CHECK(memset_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                                0, context_->GetRawTilingData()->GetCapacity()) != EOK,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_, "fail to memset tiling data"), return;);
}

ge::graphStatus WeightQuantBatchMatmulV2CustomNzSplitK::PostTiling()
{
    OP_LOGD(opName_, "final tiling data size: %zu", tilingData_->GetDataSize());

    OP_TILING_CHECK(tilingData_->GetDataSize() % sizeof(uint64_t) != 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_, "tiling data size[%zu] not aligned to 8",
                                                    tilingData_->GetDataSize()),
                    return ge::GRAPH_FAILED);

    context_->GetRawTilingData()->SetDataSize(tilingData_->GetDataSize());

    uint32_t usedAivNum = compileInfoPtr_->aicNum * 2;
    context_->SetBlockDim(CalcTschBlockDim(usedAivNum, compileInfoPtr_->aicNum, compileInfoPtr_->aivNum));
    return ge::GRAPH_SUCCESS;
}

bool WeightQuantBatchMatmulV2CustomNzSplitK::IsCapable()
{
    OP_LOGI(opName_, "Start to check Custom SplitK template");
    OP_TILING_CHECK(matmulInfoPtr_->cDtype == ge::DT_INT8 || matmulInfoPtr_->antiQuantScaleDtype == ge::DT_UINT64 ||
        matmulInfoPtr_->antiQuantScaleDtype == ge::DT_INT64,
        OP_LOGI(opName_, "Custom splitK not support quant or antiquant uint64"),
        return false);
    OP_TILING_CHECK(matmulInfoPtr_->antiQuantType != QuantType::PER_CHANNEL,
        OP_LOGI(opName_, "Custom SplitK only support per-channel"),
        return false);
    OP_TILING_CHECK(matmulInfoPtr_->transA || !matmulInfoPtr_->transB,
        OP_LOGI(opName_, "Custom SplitK only support not trans_a and trans_b"),
        return false);
    OP_TILING_CHECK(matmulInfoPtr_->mSize < M_MIN_LIMIT || matmulInfoPtr_->mSize > M_MAX_LIMIT ||
        matmulInfoPtr_->kSize > MAX_SHAPE_DIM || matmulInfoPtr_->nSize > MAX_SHAPE_DIM,
        OP_LOGI(opName_, "Custom SplitK only support 64 < m <= 256 and n < 65536 and k < 65536"),
        return false);
    WhiteListShape shape({1, matmulInfoPtr_->kSize, matmulInfoPtr_->nSize, matmulInfoPtr_->hasBias,
        matmulInfoPtr_->transA, matmulInfoPtr_->transB, 1});
    OP_TILING_CHECK(NETWORK_UNALIGN_WHITE_LIST.find(shape) == NETWORK_UNALIGN_WHITE_LIST.end() &&
        (matmulInfoPtr_->nSize % SHAPE_ALIGNED_FACTOR != 0 || matmulInfoPtr_->kSize % SHAPE_ALIGNED_FACTOR != 0),
        OP_LOGI(opName_, "Custom SplitK only support n aligned to 64 and k aligned to 64"),
        return false);
    OP_TILING_CHECK(matmulInfoPtr_->bFormat != ge::FORMAT_FRACTAL_NZ,
        OP_LOGI(opName_, "Custom SplitK only support weightNz format"),
        return false);
    OP_LOGI(opName_, "Custom SplitK check success");
    return true;
}

ge::graphStatus WeightQuantBatchMatmulV2CustomNzSplitK::InstantiateTilingData()
{
    if (tilingData_ == nullptr) {
        tilingData_ = std::unique_ptr<WeightQuantBatchMatmulV2CustomNzSplitKTilingData>(
            new (std::nothrow) WeightQuantBatchMatmulV2CustomNzSplitKTilingData());
    }
    OP_TILING_CHECK(tilingData_ == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_, "failed to instantiate tilingData"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        context_->GetRawTilingData()->GetCapacity() < tilingData_->GetDataSize(),
        VECTOR_INNER_ERR_REPORT_TILIING(opName_, "tiling data capacity %zu < actual tiling data size %zu",
                                        context_->GetRawTilingData()->GetCapacity(), tilingData_->GetDataSize()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantBatchMatmulV2CustomNzSplitK::DoOpTiling()
{
    OP_TILING_CHECK(InstantiateTilingData() == ge::GRAPH_FAILED,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_, "unable to get pointer of tiling data"),
                    return ge::GRAPH_FAILED);
    tilingData_->SetDataPtr(context_->GetRawTilingData()->GetData());
    tilingData_->set_kSize(matmulInfoPtr_->kSize);
    tilingData_->set_nSize(matmulInfoPtr_->nSize);
    tilingData_->set_mSize(matmulInfoPtr_->mSize);
    tilingData_->set_nSizeAlign(ops::CeilAlign(matmulInfoPtr_->nSize, static_cast<uint64_t>(BLOCK_CUBE)));
    tilingData_->set_kSizeAlign(ops::CeilAlign(matmulInfoPtr_->kSize,
        GetBlockAlignSizeByDataType(matmulInfoPtr_->bDtype)));
    tilingData_->set_hasBias(matmulInfoPtr_->hasBias);
    GetMatMulTiling();
    tilingData_->set_vecBlockDimN(tilingData_->get_cubeBlockDimN() * VEC_CUBE_RATIO);
    tilingData_->set_vecBlockDimK(tilingData_->get_cubeBlockDimK());
    tilingData_->set_singleK(SINGLE_K);
    tilingData_->set_vecSingleN(VECTOR_SINGLE_N);
    tilingData_->set_singleCoreKLoop(ops::CeilDiv(tilingData_->get_singleCoreK(),
                                                        static_cast<uint64_t>(tilingData_->get_singleK())));
    tilingData_->set_vectorSingleCoreN(tilingData_->get_cubeSingleCoreN() / VEC_CUBE_RATIO);
    tilingData_->set_vectorSingleCoreNTail(tilingData_->get_cubeSingleCoreNTail() / VEC_CUBE_RATIO);
    tilingData_->set_vecSingleCoreNLoop(ops::CeilDiv(tilingData_->get_vectorSingleCoreN(),
                                                        static_cast<uint64_t>(tilingData_->get_vecSingleN())));
    tilingData_->set_vecSingleCoreNTailLoop(ops::CeilDiv(tilingData_->get_vectorSingleCoreNTail(),
                                                            static_cast<uint64_t>(tilingData_->get_vecSingleN())));
    tilingData_->set_singleCoreKTailLoop(ops::CeilDiv(tilingData_->get_singleCoreKTail(),
                                                            static_cast<uint64_t>(tilingData_->get_singleK())));
    uint64_t usedAivNum = compileInfoPtr_->aivNum;
    uint64_t postSingleCoreN = ops::CeilAlign(ops::CeilDiv(matmulInfoPtr_->nSize, usedAivNum), SHAPE_ALIGNED_FACTOR);
    uint64_t postSingleN = std::min(1024UL, postSingleCoreN);
    // 160K空间用于类型转换，输入db, 64K, 64K, 32K
    uint32_t postSingleM = std::min(matmulInfoPtr_->mSize, 16 * 1024 / postSingleN);
    tilingData_->set_postSingleN(postSingleN);
    tilingData_->set_postSingleM(postSingleM);
    tilingData_->set_postSingleCoreN(postSingleCoreN);
    return ge::GRAPH_SUCCESS;
}

// 4、计算高阶API的TilingData
ge::graphStatus WeightQuantBatchMatmulV2CustomNzSplitK::DoLibApiTiling()  { return ge::GRAPH_SUCCESS; }

// 5、计算TilingKey
uint64_t WeightQuantBatchMatmulV2CustomNzSplitK::GetTilingKey() const
{
    TilingKeyConfigure tilingKeyConfigure;
    // 平台类型占2位(平台大类， 平台小类)，平台大类在高位，需要乘10
    tilingKeyConfigure.socVersionType = static_cast<uint8_t>(SocVersionType::SUPPORT_L0C_TO_OUT) * 10;
    tilingKeyConfigure.quantizationScenario = static_cast<uint8_t>(QuantizationScenario::DEFAULT);
    // 算法类型占2位(算法大类，算法小类)，算法大类在高位，需要乘10
    tilingKeyConfigure.algorithm = static_cast<uint8_t>(OptimizationAlgorithmCategory::VECTOR_ANTIQUANT) * 10 +
        static_cast<uint8_t>(OptimizationAlgorithmSubCategory::SPLIT_K);
    tilingKeyConfigure.transposeSituation =
        (static_cast<uint16_t>(matmulInfoPtr_->transA) << 1) | static_cast<uint16_t>(matmulInfoPtr_->transB);
    tilingKeyConfigure.antiquantType = static_cast<uint8_t>(matmulInfoPtr_->antiQuantType);
    tilingKeyConfigure.quantType = static_cast<uint8_t>(QuantType::NONE);
    tilingKeyConfigure.optionInputSituation = (static_cast<uint16_t>(matmulInfoPtr_->hasAntiQuantOffset) << 1);
    tilingKeyConfigure.weightFormat = static_cast<uint8_t>(WeightFormat::FRACTAL_NZ);

    tilingKeyConfigure.templateCustom = static_cast<uint8_t>(
        al1FullLoad_ ? CustomSplitKConfiguration::A_MK_FULL_LOAD : CustomSplitKConfiguration::A_NORMAL_LOAD);
    tilingKeyConfigure.apiConstexpr = 0;
    return tilingKeyConfigure.GenTilingKey();
}

// 6、计算Workspace 大小
ge::graphStatus WeightQuantBatchMatmulV2CustomNzSplitK::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opName_, "failed to get workspace size"),
                    return ge::GRAPH_FAILED);
    uint64_t workspaceSize =
        CACHE_NUM * CUBE_BASE_N * SINGLE_K * compileInfoPtr_->aicNum * GetSizeByDataType(matmulInfoPtr_->aDtype) +
        sizeof(float) * matmulInfoPtr_->mSize * matmulInfoPtr_->nSize + compileInfoPtr_->workspaceNum;

    workspaces[0] = std::max(workspaceSize, L2_SIZE);
    return ge::GRAPH_SUCCESS;
}

void WeightQuantBatchMatmulV2CustomNzSplitK::GetMatMulTiling()
{
    uint64_t singleCoreK = DEFAULT_SINGLE_CORE_SIZE;
    uint64_t cubeBlockDimK =
        std::min(ops::CeilDiv(tilingData_->get_kSizeAlign(), singleCoreK), static_cast<uint64_t>(compileInfoPtr_->aicNum));
    singleCoreK = ops::CeilAlign(ops::CeilDiv(tilingData_->get_kSizeAlign(), cubeBlockDimK), DEFAULT_SINGLE_CORE_SIZE);
    cubeBlockDimK = ops::CeilDiv(tilingData_->get_kSizeAlign(), singleCoreK);
    // L1的一半空间256K用来载入A矩阵
    if (matmulInfoPtr_->mSize * singleCoreK * GetSizeByDataType(matmulInfoPtr_->aDtype) <= 256 * 1024) {
        al1FullLoad_ = true;
    }
    uint64_t cubeBlockDimN = compileInfoPtr_->aicNum / cubeBlockDimK;
    uint64_t cubeSingleCoreN = ops::CeilAlign(ops::CeilDiv(tilingData_->get_nSizeAlign(), cubeBlockDimN), SHAPE_ALIGNED_FACTOR);
    cubeBlockDimN = ops::CeilDiv(tilingData_->get_nSizeAlign(), cubeSingleCoreN);
    if (cubeSingleCoreN < cubeSingleN_) {
        cubeSingleN_ = cubeSingleN_ >> 1;
    }
    tilingData_->set_cubeSingleCoreN(cubeSingleCoreN);
    tilingData_->set_singleCoreK(singleCoreK);
    tilingData_->set_cubeBlockDimK(cubeBlockDimK);
    tilingData_->set_cubeBlockDimN(cubeBlockDimN);
    tilingData_->set_cubeSingleM(
        ops::CeilAlign(std::min(matmulInfoPtr_->mSize, CUBE_BASE_M), static_cast<uint64_t>(AscendC::BLOCK_CUBE)));
    tilingData_->set_cubeSingleN(cubeSingleN_);
    tilingData_->set_cubeBaseK(CUBE_BASE_K);
    tilingData_->set_cubeSingleCoreNLoop(ops::CeilDiv(cubeSingleCoreN, cubeSingleN_));
    uint64_t usedAicNum = ops::CeilDiv(tilingData_->get_nSizeAlign(), cubeSingleCoreN);
    uint64_t cubeSingleCoreNTail = tilingData_->get_nSizeAlign() - (usedAicNum - 1) * cubeSingleCoreN;
    tilingData_->set_cubeSingleCoreNTail(cubeSingleCoreNTail);
    uint64_t cubeSingleCoreNOriTail = tilingData_->get_nSize() -  (usedAicNum - 1) * cubeSingleCoreN;
    usedAicNum = ops::CeilDiv(tilingData_->get_kSizeAlign(), singleCoreK);
    uint64_t singleCoreKTail = tilingData_->get_kSizeAlign() - (usedAicNum - 1) * singleCoreK;
    tilingData_->set_cubeSingleCoreNLoop(ops::CeilDiv(cubeSingleCoreN, cubeSingleN_));
    tilingData_->set_cubeSingleCoreNOriTail(cubeSingleCoreNOriTail);
    uint64_t singleCoreKOriTail = tilingData_->get_kSize() - (usedAicNum - 1) * singleCoreK;
    tilingData_->set_singleCoreKTail(singleCoreKTail);
    tilingData_->set_singleCoreKOriTail(singleCoreKOriTail);
    tilingData_->set_cubeSingleCoreNTailLoop(ops::CeilDiv(cubeSingleCoreNTail,
                                                            static_cast<uint64_t>(cubeSingleN_)));
}

}  // namespace optiling

