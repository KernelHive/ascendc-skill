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
 * \file weight_quant_batch_matmul_v2_tiling_splitk.cpp
 * \brief
 */

#include "weight_quant_batch_matmul_v2_tiling_splitk.h"

#include "weight_quant_batch_matmul_v2_compute_matmul_tiling.h"
#include "weight_quant_batch_matmul_v2_white_list.h"

namespace optiling {

const std::set<WhiteListShape> MIX_SPLIT_K_WHITE_LIST = {
    // JYXC
    {24, 12288, 1792, false, false, false, 1},
    {24, 12288, 7808, false, false, false, 1},
    {24, 3904, 12288, false, false, false, 1},
    {24, 1536, 12288, false, false, false, 1}};

void WeightQuantBatchMatmulV2TilingSplitK::Reset()
{
    WeightQuantBatchMatmulV2Tiling::Reset();
    OP_TILING_CHECK(memset_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(), 0,
                             context_->GetRawTilingData()->GetCapacity()) != EOK,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_, "fail to memset tiling data"), return;);
}

/*
The function is limite of splitK
1. bf16*int8=>bf16 without antiquantoffset
2. JYXC white case in pergroup and weightND
*/
bool WeightQuantBatchMatmulV2TilingSplitK::IsCapable()
{
    OP_LOGI(opName_, "Begin check SplitK");
    // 当前仅支持bf16*int8=>bf16
    if (matmulInfoPtr_->aDtype != ge::DT_BF16 || matmulInfoPtr_->bDtype != ge::DT_INT8 ||
        matmulInfoPtr_->cDtype != ge::DT_BF16) {
        OP_LOGI(opName_,
                "SplitK only support bf16*int8=bf16, current aDtype: [%s], bDtype: [%s], cDtype: [%s]",
                ge::TypeUtils::DataTypeToAscendString(matmulInfoPtr_->aDtype).GetString(),
                ge::TypeUtils::DataTypeToAscendString(matmulInfoPtr_->bDtype).GetString(),
                ge::TypeUtils::DataTypeToAscendString(matmulInfoPtr_->cDtype).GetString());
        return false;
    }
    OP_TILING_CHECK(matmulInfoPtr_->antiQuantScaleDtype == ge::DT_UINT64 ||
                    matmulInfoPtr_->antiQuantScaleDtype == ge::DT_INT64,
                    OP_LOGI(opName_, "SplitK done not support antiquant scale dtype is uint64 or int64"),
                    return false);
    // only support jyxc case
    if (matmulInfoPtr_->antiQuantType == QuantType::PER_GROUP && matmulInfoPtr_->bFormat != ge::FORMAT_FRACTAL_NZ) {
        WhiteListShape shape({matmulInfoPtr_->mSize, matmulInfoPtr_->kSize, matmulInfoPtr_->nSize,
                              matmulInfoPtr_->hasBias, matmulInfoPtr_->transA, matmulInfoPtr_->transB, 1});
        OP_TILING_CHECK(MIX_SPLIT_K_WHITE_LIST.find(shape) == MIX_SPLIT_K_WHITE_LIST.end(),
                        OP_LOGI(opName_, "the case is not match white case for split k"), return false);
        OP_TILING_CHECK(!matmulInfoPtr_->hasAntiQuantOffset,
                        OP_LOGI(opName_, "the white case must with antiquant offset"), return false);
        OP_LOGI(opName_, "Check SplitK succ");
        return true;
    }
    return false;
}

bool WeightQuantBatchMatmulV2TilingSplitK::GetMatMulTiling()
{
    matmul_tiling::DataType mmInputDtype = GetMatmulTilingDtype(matmulInfoPtr_->aDtype);
    matmul_tiling::MatmulApiTiling mmTiling;
    mmTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmInputDtype,
                        matmulInfoPtr_->transA);
    mmTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmInputDtype,
                        matmulInfoPtr_->transB);
    mmTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                      GetMatmulTilingDtype(ge::DT_FLOAT));
    mmTiling.SetBias(matmulInfoPtr_->hasBias);
    mmTiling.SetOrgShape(matmulInfoPtr_->mSize, matmulInfoPtr_->nSize, matmulInfoPtr_->kSize);

    uint64_t cubeSingleCoreN = 1024;
    uint64_t cubeSingleCoreK = 64;
    mmTiling.SetShape(matmulInfoPtr_->mSize, cubeSingleCoreN, cubeSingleCoreK);
    mmTiling.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize);
    mmTiling.SetFixSplit(ops::CeilAlign(static_cast<uint32_t>(matmulInfoPtr_->mSize), BLOCK_CUBE), BASIC_BLOCK,
                            cubeSingleCoreK);
    OP_TILING_CHECK(mmTiling.GetTiling(tilingData_->matmulTiling) == -1,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_, "failed to get matmul tiling"),
                    return false);
    tilingData_->matmulTiling.set_shareL1Size(0);
    tilingData_->matmulTiling.set_dbL0C(2);  // 2: db on

    return true;
}

ge::graphStatus WeightQuantBatchMatmulV2TilingSplitK::DoOpTiling()
{
   OP_TILING_CHECK(InstantiateTilingData() == ge::GRAPH_FAILED,
                   VECTOR_INNER_ERR_REPORT_TILIING(opName_, "Unable to get pointer of tiling data"),
                   return ge::GRAPH_FAILED);

    tilingData_->set_groupSize(matmulInfoPtr_->groupSize);
    uint64_t weightBlockAlignSize = GetBlockAlignSizeByDataType(matmulInfoPtr_->bDtype);
    if (matmulInfoPtr_->transB) {
        tilingData_->set_kAlign(ops::CeilAlign(matmulInfoPtr_->kSize, weightBlockAlignSize));
        tilingData_->set_nAlign(matmulInfoPtr_->nSize);
        tilingData_->set_kPadSize(static_cast<uint8_t>(tilingData_->get_kAlign() - matmulInfoPtr_->kSize));
    } else {
        tilingData_->set_kAlign(matmulInfoPtr_->kSize);
        tilingData_->set_nAlign(ops::CeilAlign(matmulInfoPtr_->nSize, weightBlockAlignSize));
        tilingData_->set_nPadSize(static_cast<uint8_t>(tilingData_->get_nAlign() - matmulInfoPtr_->nSize));
    }
    tilingData_->set_kSize(matmulInfoPtr_->kSize);
    tilingData_->set_nSize(matmulInfoPtr_->nSize);
    tilingData_->set_mSize(matmulInfoPtr_->mSize);

    // 非转置场景重新实现切分逻辑
    // n方向以1024为最小划分单元
    uint64_t vecSingleN = 512;
    uint64_t nFactor = ops::CeilDiv(matmulInfoPtr_->nSize, vecSingleN * 2);
    uint64_t usedCoreNumMaxResult = 0;
    for (uint64_t cubeDimN = nFactor; cubeDimN >= 1; cubeDimN--) {
        if (nFactor % cubeDimN != 0) {
            continue;
        }
        uint64_t cubeDimK = compileInfoPtr_->aicNum / cubeDimN;
        uint64_t usedCoreNum = cubeDimN * cubeDimK;
        if (usedCoreNum > usedCoreNumMaxResult) {
            tilingData_->set_cubeBlockDimK(static_cast<uint8_t>(cubeDimK));
            tilingData_->set_cubeBlockDimN(static_cast<uint8_t>(cubeDimN));
            usedCoreNumMaxResult = usedCoreNum;
        }
    }
    tilingData_->set_cubeBlockDimM(1);
    tilingData_->set_vecBlockDimK(tilingData_->get_cubeBlockDimK());
    tilingData_->set_vecSingleK(static_cast<uint32_t>(ops::CeilAlign(
        ops::CeilDiv(matmulInfoPtr_->kSize, static_cast<uint64_t>(tilingData_->get_cubeBlockDimK())),
        static_cast<uint64_t>(matmulInfoPtr_->groupSize))));
    tilingData_->set_vecSingleN(static_cast<uint32_t>(vecSingleN));
    // vec固定保持2倍cube的方式切分
    tilingData_->set_vecBlockDimN(tilingData_->get_cubeBlockDimN() * 2);
    tilingData_->set_vecSingleKGroupNum(
        ops::CeilDiv(static_cast<uint64_t>(tilingData_->get_vecSingleK()), matmulInfoPtr_->groupSize));
    OP_TILING_CHECK(
        !GetMatMulTiling(),
        VECTOR_INNER_ERR_REPORT_TILIING(opName_, "Failed to get mm tiling for mnk[%ld, %ld, %ld]",
                                        matmulInfoPtr_->mSize, matmulInfoPtr_->nSize, matmulInfoPtr_->kSize),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantBatchMatmulV2TilingSplitK::InstantiateTilingData()
{
    if (tilingData_ == nullptr) {
        tilingData_ = std::unique_ptr<WeightQuantBatchMatmulV2TilingData>(
            new (std::nothrow) WeightQuantBatchMatmulV2TilingData());
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

ge::graphStatus WeightQuantBatchMatmulV2TilingSplitK::DoLibApiTiling()
{
    tilingData_->set_cubeSingleNTailLoop(
        ops::CeilDiv(matmulInfoPtr_->nSize % tilingData_->matmulTiling.get_singleCoreN(),
                        static_cast<uint64_t>(tilingData_->matmulTiling.get_singleCoreN())));
    tilingData_->set_cubeTailM(
        CalcTailSize(matmulInfoPtr_->mSize, static_cast<uint64_t>(tilingData_->matmulTiling.get_singleCoreM())));
    tilingData_->set_cubeTailN(
        CalcTailSize(matmulInfoPtr_->nSize, static_cast<uint64_t>(tilingData_->matmulTiling.get_baseN())));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantBatchMatmulV2TilingSplitK::GetWorkspaceSize()
{
    uint64_t weightWorkspacesNum = 4;
    uint64_t nF16AlignTo512bSize = ops::CeilDiv(tilingData_->get_nSize(), 256UL) * 256;
    uint64_t weightCacheLen = weightWorkspacesNum * tilingData_->get_cubeBlockDimK() *
             matmulInfoPtr_->groupSize * nF16AlignTo512bSize;
    uint64_t mmResultCache = tilingData_->get_nSize() * tilingData_->get_mSize();
    workspaceSize_ = weightCacheLen * sizeof(matmulInfoPtr_->aDtype) + mmResultCache * sizeof(float) +
                     compileInfoPtr_->workspaceNum;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantBatchMatmulV2TilingSplitK::PostTiling()
{
    OP_LOGD(opName_, "final tiling data size: %zu", tilingData_->GetDataSize());

    OP_TILING_CHECK(tilingData_->GetDataSize() % sizeof(uint64_t) != 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_, "tiling data size[%zu] not aligned to 8",
                                                    tilingData_->GetDataSize()),
                    return ge::GRAPH_FAILED);
    context_->GetRawTilingData()->SetDataSize(tilingData_->GetDataSize());
    uint32_t usedAicNum = tilingData_->get_cubeBlockDimM() * tilingData_->get_cubeBlockDimN();
    uint32_t usedAivNum = tilingData_->get_vecBlockDimK() * tilingData_->get_vecBlockDimN();
    context_->SetBlockDim(std::max(usedAicNum, CalcTschBlockDim(
        usedAivNum, compileInfoPtr_->aicNum, compileInfoPtr_->aivNum)));
    size_t *workspaces = context_->GetWorkspaceSizes(1);  // set workspace
    workspaces[0] = workspaceSize_;

    tilingData_->SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    return ge::GRAPH_SUCCESS;
}

uint64_t WeightQuantBatchMatmulV2TilingSplitK::GetTilingKey() const
{
    return RecursiveSum(matmulInfoPtr_->transA, matmulInfoPtr_->transB, matmulInfoPtr_->antiQuantType,
                        matmulInfoPtr_->hasAntiQuantOffset, matmulInfoPtr_->quantType,
                        KernelTemplateType::MIX_SPLIT_K);
}

}
