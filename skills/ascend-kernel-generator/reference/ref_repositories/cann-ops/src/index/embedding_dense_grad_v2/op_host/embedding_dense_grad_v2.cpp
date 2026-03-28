/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file embedding_dense_grad_v2.cpp
 */
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_info.h"
#include "tiling/tiling_api.h"
#include "embedding_dense_grad_v2_tiling.h"

// tools api
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
namespace ops {
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    std::printf("nullptr error!");                                                               \
    return ge::GRAPH_FAILED;                                                                     \
  }
}  // namespace ops
namespace optiling {
#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)
}  // namespace optiling

using namespace ge;

namespace optiling {
    constexpr uint64_t RecursiveSum()
    {
        return 0;
    }

    template <typename T, typename... Args> constexpr uint64_t RecursiveSum(T templateId, Args... templateIds)
    {
        return static_cast<uint64_t>(templateId) + 10 * RecursiveSum(templateIds...);
    }

    constexpr uint32_t VEC_PROCESS_SIZE = 256;
    constexpr uint32_t SIZE_OF_FP32 = 4;
    constexpr uint32_t BLOCK_SIZE = 32;
    constexpr uint32_t RESERVED_UB_SIZE = 20480;
    constexpr uint32_t USE_IDX_NUM_IN_UB = 3;
    constexpr uint32_t USE_GRAD_NUM_IN_UB = 3;
    
    constexpr uint32_t SMALL_DIM_THRESHOLD = 512;
    constexpr uint32_t CAST_MAX_NUM = 16777216;
    constexpr uint32_t ALIGN_32_NUM_SPCAE = 300;
    constexpr uint32_t UB_SORT_PART = 10;

    class EmbeddingDenseGradV2Tiling {
    public:
        explicit EmbeddingDenseGradV2Tiling(gert::TilingContext* context) : tilingContext_(context) {}
        ge::graphStatus Init();
        ge::graphStatus SetKernelTiling();
        void TilingDataPrint() const;
    private:
        inline void SetTilingKeyMode();
        inline void BaseTiling(const int64_t gradRow);
        inline void Tiling4Scale();
        inline void Tiling4Deterministic(const int64_t gradRow);
        inline void CalMaxFormerNum(int64_t ubSizeLeft);
        inline void Tiling4SmallDim(const int64_t gradRow);
        inline bool CheckIsSmallDim(const int64_t gradLastDim);

        EmbeddingDenseGradV2TilingData tilingData_;
        gert::TilingContext* tilingContext_ = nullptr;
        uint64_t coreNum_ = 0;
        uint64_t numWeights_ = 0;
        uint64_t embeddingDim_ = 0;
        uint64_t paddingIdx_ = 0;
        uint64_t ubSize_ = 0;
        bool scaleGrad_ = false;
    private:
        uint64_t formerRow_ = 0;
        uint64_t tailRow_ = 0;
        uint64_t formerRowRepTime_ = 0;
        uint64_t computeMask_ = 0;
        uint64_t formerComputeRepTime_ = 0;
        uint64_t formerComputeFormerNum_ = 0;
        uint64_t formerComputeTailNum_ = 0;
        uint64_t tailComputeRepTime_ = 0;
        uint64_t tailComputeFormerNum_ = 0;
        uint64_t tailComputeTailNum_ = 0;
    private:
        // small dim
        uint64_t partNum_ = 0;
        uint64_t formerCopyRow_ = 0;
        uint64_t tailCopyRow_ = 0;
        uint64_t formerCopyTime_ = 0;
        uint64_t tailCopyTime_ = 0;
        uint64_t maxRowInUb_ = 0;
        uint64_t formerLastRow_ = 0;
        uint64_t tailLastRow_ = 0;
    private:
        // scale
        uint64_t tailCoreRowNum_ = 0;
        uint64_t formerCoreRowNum_ = 0;
        uint64_t scaleMask_ = 0;
        uint64_t scaleRepTime_ = 0;
        uint64_t scaleFormerNum_ = 0;
        uint64_t scaleTailNum_ = 0;
        uint64_t formerCoreRowRepTime_ = 0;
    private:
        // determin
        bool isDeterministMode_ = false;
        uint64_t tailRowNum_ = 0;
        uint64_t formerRowNum_ = 0;
        uint64_t formerRowNumRepTime_ = 0;
        uint64_t determinMask_ = 0;
        uint64_t formerDeterminRepTime_ = 0;
        uint64_t formerDeterminFormerNum_ = 0;
        uint64_t formerDeterminTailNum_ = 0;
        uint64_t tailDeterminRepTime_ = 0;
        uint64_t tailDeterminFormerNum_ = 0;
        uint64_t tailDeterminTailNum_ = 0;
        uint64_t gradRow_ = 0;

    private:
        // big shape
        uint64_t tailEmbeddingDim_ = 0;
        uint64_t formerEmbeddingDim_ = 0;
        uint64_t formerDimRepTime_ = 0;
        uint64_t maxFormerNum = 0; // max formerEmbeddingDim_ that ub can calculate
    };

    inline void EmbeddingDenseGradV2Tiling::SetTilingKeyMode()
    { 
        bool isSmallDim = CheckIsSmallDim(embeddingDim_);
        uint64_t tilingKey = RecursiveSum(scaleGrad_, isDeterministMode_, isSmallDim);
        tilingContext_->SetTilingKey(tilingKey);
    }

    inline void EmbeddingDenseGradV2Tiling::Tiling4Scale()
    {
        OP_LOGD(tilingContext_->GetNodeName(), "scaleTiling start");
        tailCoreRowNum_ = numWeights_ / coreNum_;
        formerCoreRowNum_ = tailCoreRowNum_ + 1;
        formerCoreRowRepTime_ = numWeights_ % coreNum_;
        scaleMask_ = VEC_PROCESS_SIZE / SIZE_OF_FP32;
        
        // formerDim compute params
        formerComputeRepTime_ = formerEmbeddingDim_ / scaleMask_;
        formerComputeFormerNum_ = formerComputeRepTime_ * scaleMask_;
        formerComputeTailNum_ = formerEmbeddingDim_ - formerComputeFormerNum_;

        // tailDim compute params
        tailComputeRepTime_ = tailEmbeddingDim_ / scaleMask_;
        tailComputeFormerNum_ = tailComputeRepTime_ * scaleMask_;
        tailComputeTailNum_ = tailEmbeddingDim_ - tailComputeFormerNum_;
        OP_LOGD(tilingContext_->GetNodeName(), "scaleTiling finish");
    }

    inline void EmbeddingDenseGradV2Tiling::BaseTiling(const int64_t gradRow)
    {
        OP_LOGD(tilingContext_->GetNodeName(), "BaseTiling start");
        gradRow_ = gradRow;
        tailRow_ = gradRow / coreNum_;
        formerRow_ = tailRow_ + 1;
        formerRowRepTime_ = gradRow % coreNum_;

        formerEmbeddingDim_ = embeddingDim_ <= maxFormerNum ? embeddingDim_ : maxFormerNum;
        formerDimRepTime_ = embeddingDim_ / formerEmbeddingDim_;
        tailEmbeddingDim_ = embeddingDim_ - formerEmbeddingDim_ * formerDimRepTime_;
        
        // formerDim compute params
        computeMask_ = VEC_PROCESS_SIZE / SIZE_OF_FP32;
        formerComputeRepTime_ = formerEmbeddingDim_ / computeMask_;
        formerComputeFormerNum_ = formerComputeRepTime_ * computeMask_;
        formerComputeTailNum_ = formerEmbeddingDim_ - formerComputeFormerNum_;
        // tailDim compute params
        tailComputeRepTime_ = tailEmbeddingDim_ / computeMask_;
        tailComputeFormerNum_ = tailComputeRepTime_ * computeMask_;
        tailComputeTailNum_ = tailEmbeddingDim_ - tailComputeFormerNum_;

        OP_LOGD(tilingContext_->GetNodeName(), "BaseTiling finish");
        if (scaleGrad_) {
            Tiling4Scale();
        }
    }

    inline void EmbeddingDenseGradV2Tiling::Tiling4Deterministic(const int64_t gradRow)
    {
        OP_LOGD(tilingContext_->GetNodeName(), "determinist tiling start");
        gradRow_ = gradRow;
        tailRowNum_ = gradRow / coreNum_;
        formerRowNum_ = tailRowNum_ + 1;
        formerRowNumRepTime_ = gradRow % coreNum_;

        formerEmbeddingDim_ = embeddingDim_ <= maxFormerNum ? embeddingDim_ : maxFormerNum;
        formerDimRepTime_ = embeddingDim_ / formerEmbeddingDim_;
        tailEmbeddingDim_ = embeddingDim_ - formerEmbeddingDim_ * formerDimRepTime_;
        
        // formerDim determin compute params
        determinMask_ = VEC_PROCESS_SIZE / SIZE_OF_FP32;
        formerDeterminRepTime_ = formerEmbeddingDim_ / determinMask_;
        formerDeterminFormerNum_ = formerDeterminRepTime_ * determinMask_;
        formerDeterminTailNum_ = formerEmbeddingDim_ - formerDeterminFormerNum_;
        // tailDim determin compute params
        tailDeterminRepTime_ = tailEmbeddingDim_ / determinMask_;
        tailDeterminFormerNum_ = tailDeterminRepTime_ * determinMask_;
        tailDeterminTailNum_ = tailEmbeddingDim_ - tailDeterminFormerNum_;

        OP_LOGD(tilingContext_->GetNodeName(), "determinist tiling finish");
        if (scaleGrad_) {
            Tiling4Scale();
        }
    }

    inline void EmbeddingDenseGradV2Tiling::CalMaxFormerNum(int64_t ubSizeLeft)
    {
        auto const gradDtype = tilingContext_->GetInputDesc(0)->GetDataType();
        uint64_t idxAlignNum = BLOCK_SIZE / sizeof(int);
        uint64_t gradAlignNum = BLOCK_SIZE / sizeof(gradDtype);
        ubSizeLeft -= RESERVED_UB_SIZE + idxAlignNum * sizeof(int) * USE_IDX_NUM_IN_UB;
        uint64_t availableUbForGrad = ubSizeLeft > 0 ? ubSizeLeft : 0;
        maxFormerNum = (availableUbForGrad / (gradAlignNum * sizeof(gradDtype) * USE_GRAD_NUM_IN_UB)) * gradAlignNum;
    }

    inline void EmbeddingDenseGradV2Tiling::Tiling4SmallDim(const int64_t gradRow)
    {
        OP_LOGD(tilingContext_->GetNodeName(), "small dim tiling start");
        formerEmbeddingDim_ = embeddingDim_ <= maxFormerNum ? embeddingDim_ : maxFormerNum;
        formerDimRepTime_ = embeddingDim_ / formerEmbeddingDim_;
        tailEmbeddingDim_ = embeddingDim_ - formerEmbeddingDim_ * formerDimRepTime_;
        partNum_ = 1;
        formerCopyRow_ = gradRow / coreNum_;
        tailCopyRow_ = gradRow - formerCopyRow_ * (coreNum_ - 1);
        uint64_t dataInBlock = BLOCK_SIZE / sizeof(float);
        uint64_t divNum = UB_SORT_PART + (embeddingDim_ + dataInBlock - 1) / dataInBlock * dataInBlock;
        maxRowInUb_ = ((ubSize_ - RESERVED_UB_SIZE) / SIZE_OF_FP32 - embeddingDim_ - VEC_PROCESS_SIZE) / divNum;
        formerCopyTime_ = (formerCopyRow_ + maxRowInUb_ - 1) / maxRowInUb_;
        tailCopyTime_ = (tailCopyRow_ + maxRowInUb_ - 1) / maxRowInUb_;
        formerLastRow_ = formerCopyRow_ - maxRowInUb_ * (formerCopyTime_ - 1);
        tailLastRow_ = tailCopyRow_ - maxRowInUb_ * (tailCopyTime_ - 1);
        OP_LOGD(tilingContext_->GetNodeName(), "small dim tiling end");
        if (scaleGrad_) {
            Tiling4Scale();
        }
    }

    ge::graphStatus EmbeddingDenseGradV2Tiling::Init()
    {
        OP_LOGD(tilingContext_->GetNodeName(), "Tiling initing");
        isDeterministMode_ = (tilingContext_->GetDeterministic() == 1);
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext_->GetPlatformInfo());
        auto attrs = tilingContext_->GetAttrs();
        auto selfShape = tilingContext_->GetInputShape(0)->GetStorageShape();
        int64_t gradRow = 1;
        for (size_t i = 0; i < selfShape.GetDimNum() - 1; i++) {
            gradRow *= selfShape.GetDim(i);
        }
        embeddingDim_ = selfShape.GetDim(selfShape.GetDimNum() - 1);
        numWeights_ = *(attrs->GetAttrPointer<uint64_t>)(0);
        paddingIdx_ = *(attrs->GetAttrPointer<uint64_t>)(1);
        scaleGrad_  = *(attrs->GetAttrPointer<bool>)(2);
        size_t alignNum = BLOCK_SIZE / sizeof(int64_t);
        size_t scaleWorkspaceSize = ((numWeights_ + alignNum - 1) / alignNum) * alignNum * sizeof(int64_t);
        size_t sysWorkspaceSize = 16 * 1024 * 1024;
        sysWorkspaceSize = scaleGrad_ ? sysWorkspaceSize + scaleWorkspaceSize : sysWorkspaceSize;
        size_t *currentWorkSpace = tilingContext_->GetWorkspaceSizes(1);
        currentWorkSpace[0] = sysWorkspaceSize;

        coreNum_ = ascendcPlatform.GetCoreNumAiv();
        coreNum_ = scaleGrad_ ?
                    std::min(coreNum_, std::min(static_cast<uint64_t>(numWeights_), static_cast<uint64_t>(gradRow))) :
                    std::min(coreNum_, static_cast<uint64_t>(gradRow));                    
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
        if (coreNum_ == 0 || embeddingDim_ == 0) {
            OP_LOGE(tilingContext_->GetNodeName(), "coreNum %lu, embeddingDim %lu", coreNum_, embeddingDim_);
            return ge::GRAPH_FAILED;
        }
        
        CalMaxFormerNum(ubSize_);
        OP_TILING_CHECK((maxFormerNum == 0),
                         VECTOR_INNER_ERR_REPORT_TILIING(tilingContext_->GetNodeName(), "Do not have enough ub size."),
                         return ge::GRAPH_FAILED);
        SetTilingKeyMode();
        tilingContext_->SetNeedAtomic(true);
        if (isDeterministMode_) {
            Tiling4Deterministic(gradRow);
        } else if(CheckIsSmallDim(embeddingDim_)) {
            Tiling4SmallDim(gradRow);
        } else {
            BaseTiling(gradRow);
        }
        OP_LOGD(tilingContext_->GetNodeName(), "Tiling inited");
        return ge::GRAPH_SUCCESS;
    }

    bool EmbeddingDenseGradV2Tiling::CheckIsSmallDim(const int64_t gradLastDim)
    {
        return !isDeterministMode_ && gradLastDim <= SMALL_DIM_THRESHOLD &&
               numWeights_ <= CAST_MAX_NUM;
    }

    ge::graphStatus EmbeddingDenseGradV2Tiling::SetKernelTiling()
    {
        tilingContext_->SetBlockDim(coreNum_);
        tilingData_.params.set_tailRowNum(tailRow_);
        tilingData_.params.set_formerRowNum(formerRow_);
        tilingData_.params.set_formerRowRepTime(formerRowRepTime_);
        tilingData_.params.set_computeMask(computeMask_);
        tilingData_.params.set_formerComputeRepTime(formerComputeRepTime_);
        tilingData_.params.set_formerComputeFormerNum(formerComputeFormerNum_);
        tilingData_.params.set_formerComputeTailNum(formerComputeTailNum_);
        tilingData_.params.set_tailComputeRepTime(tailComputeRepTime_);
        tilingData_.params.set_tailComputeFormerNum(tailComputeFormerNum_);
        tilingData_.params.set_tailComputeTailNum(tailComputeTailNum_);
        tilingData_.params.set_embeddingDim(embeddingDim_);
        tilingData_.params.set_numWeights(numWeights_);
        tilingData_.params.set_paddingIdx(paddingIdx_);
        tilingData_.params.set_scaleGradByFreq(scaleGrad_);
        tilingData_.params.set_formerDimRepTime(formerDimRepTime_);
        tilingData_.params.set_formerEmbeddingDim(formerEmbeddingDim_);
        tilingData_.params.set_tailEmbeddingDim(tailEmbeddingDim_);

        tilingData_.scaleTiling.set_tailCoreRowNum(tailCoreRowNum_);
        tilingData_.scaleTiling.set_formerCoreRowNum(formerCoreRowNum_);
        tilingData_.scaleTiling.set_formerCoreRowRepTime(formerCoreRowRepTime_);
        tilingData_.scaleTiling.set_mask(scaleMask_);
        tilingData_.scaleTiling.set_formerComputeRepTime(formerComputeRepTime_);
        tilingData_.scaleTiling.set_formerComputeFormerNum(formerComputeFormerNum_);
        tilingData_.scaleTiling.set_formerComputeTailNum(formerComputeTailNum_);
        tilingData_.scaleTiling.set_tailComputeRepTime(tailComputeRepTime_);
        tilingData_.scaleTiling.set_tailComputeFormerNum(tailComputeFormerNum_);
        tilingData_.scaleTiling.set_tailComputeTailNum(tailComputeTailNum_);

        tilingData_.determinTiling.set_gradRow(gradRow_);
        tilingData_.determinTiling.set_tailRowNum(tailRowNum_);
        tilingData_.determinTiling.set_formerRowNum(formerRowNum_);
        tilingData_.determinTiling.set_formerRowNumRepTime(formerRowNumRepTime_);
        tilingData_.determinTiling.set_computeMask(determinMask_);
        tilingData_.determinTiling.set_formerComputeRepTime(formerDeterminRepTime_);
        tilingData_.determinTiling.set_formerComputeFormerNum(formerDeterminFormerNum_);
        tilingData_.determinTiling.set_formerComputeTailNum(formerDeterminTailNum_);
        tilingData_.determinTiling.set_tailComputeRepTime(tailDeterminRepTime_);
        tilingData_.determinTiling.set_tailComputeFormerNum(tailDeterminFormerNum_);
        tilingData_.determinTiling.set_tailComputeTailNum(tailDeterminTailNum_);

        tilingData_.smallDimTiling.set_partNum(partNum_);
        tilingData_.smallDimTiling.set_formerCopyRow(formerCopyRow_);
        tilingData_.smallDimTiling.set_tailCopyRow(tailCopyRow_);
        tilingData_.smallDimTiling.set_formerCopyTime(formerCopyTime_);
        tilingData_.smallDimTiling.set_tailCopyTime(tailCopyTime_);
        tilingData_.smallDimTiling.set_maxRowInUb(maxRowInUb_);
        tilingData_.smallDimTiling.set_formerLastRow(formerLastRow_);
        tilingData_.smallDimTiling.set_tailLastRow(tailLastRow_);

        tilingData_.SaveToBuffer(tilingContext_->GetRawTilingData()->GetData(),
                                 tilingContext_->GetRawTilingData()->GetCapacity());
        tilingContext_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
        TilingDataPrint();
        return ge::GRAPH_SUCCESS;
    }

    void EmbeddingDenseGradV2Tiling::TilingDataPrint() const
    {
        OP_LOGD(tilingContext_->GetNodeName(), "maxFormerNum:            %lu", maxFormerNum);
        OP_LOGD(tilingContext_->GetNodeName(), "coreNum:                 %lu", coreNum_);
        OP_LOGD(tilingContext_->GetNodeName(), "numWeights:              %lu", numWeights_);
        OP_LOGD(tilingContext_->GetNodeName(), "embeddingDim:            %lu", embeddingDim_);
        OP_LOGD(tilingContext_->GetNodeName(), "paddingIdx:              %lu", paddingIdx_);
        OP_LOGD(tilingContext_->GetNodeName(), "scaleGrad:               %d", scaleGrad_);
        OP_LOGD(tilingContext_->GetNodeName(), "gradRow:                 %lu", gradRow_);
        OP_LOGD(tilingContext_->GetNodeName(), "tailRow:                 %lu", tailRow_);
        OP_LOGD(tilingContext_->GetNodeName(), "formerRow:               %lu", formerRow_);
        OP_LOGD(tilingContext_->GetNodeName(), "formerRowRepTime:        %lu", formerRowRepTime_);

        OP_LOGD(tilingContext_->GetNodeName(), "formerDimRepTime:        %lu", formerDimRepTime_);
        OP_LOGD(tilingContext_->GetNodeName(), "formerEmbeddingDim:      %lu", formerEmbeddingDim_);
        OP_LOGD(tilingContext_->GetNodeName(), "tailEmbeddingDim:        %lu", tailEmbeddingDim_);

        OP_LOGD(tilingContext_->GetNodeName(), "computeMask:             %lu", computeMask_);
        OP_LOGD(tilingContext_->GetNodeName(), "formerComputeRepTime:    %lu", formerComputeRepTime_);
        OP_LOGD(tilingContext_->GetNodeName(), "formerComputeFormerNum:  %lu", formerComputeFormerNum_);
        OP_LOGD(tilingContext_->GetNodeName(), "formerComputeTailNum:    %lu", formerComputeTailNum_);

        OP_LOGD(tilingContext_->GetNodeName(), "tailComputeRepTime:      %lu", tailComputeRepTime_);
        OP_LOGD(tilingContext_->GetNodeName(), "tailComputeFormerNum:    %lu", tailComputeFormerNum_);
        OP_LOGD(tilingContext_->GetNodeName(), "tailComputeTailNum:      %lu", tailComputeTailNum_);

        OP_LOGD(tilingContext_->GetNodeName(), "tailCoreRow:             %lu", tailCoreRowNum_);
        OP_LOGD(tilingContext_->GetNodeName(), "formerCoreRow:           %lu", formerCoreRowNum_);
        OP_LOGD(tilingContext_->GetNodeName(), "formerCoreRowRepTime:    %lu", formerCoreRowRepTime_);
        OP_LOGD(tilingContext_->GetNodeName(), "scaleMask:               %lu", scaleMask_);
        OP_LOGD(tilingContext_->GetNodeName(), "scaleRepTime:            %lu", scaleRepTime_);
        OP_LOGD(tilingContext_->GetNodeName(), "scaleFormerNum:          %lu", scaleFormerNum_);
        OP_LOGD(tilingContext_->GetNodeName(), "scaleTailNum:            %lu", scaleTailNum_);

        OP_LOGD(tilingContext_->GetNodeName(), "tailRowNum:              %lu", tailRowNum_);
        OP_LOGD(tilingContext_->GetNodeName(), "formerRowNum:            %lu", formerRowNum_);
        OP_LOGD(tilingContext_->GetNodeName(), "formerRowNumRepTime:     %lu", formerRowNumRepTime_);
        OP_LOGD(tilingContext_->GetNodeName(), "determinMask:            %lu", determinMask_);
        OP_LOGD(tilingContext_->GetNodeName(), "formerDeterminRepTime:   %lu", formerDeterminRepTime_);
        OP_LOGD(tilingContext_->GetNodeName(), "formerDeterminFormerNum: %lu", formerDeterminFormerNum_);
        OP_LOGD(tilingContext_->GetNodeName(), "formerDeterminTailNum:   %lu", formerDeterminTailNum_);

        OP_LOGD(tilingContext_->GetNodeName(), "tailDeterminRepTime:     %lu", tailDeterminRepTime_);
        OP_LOGD(tilingContext_->GetNodeName(), "tailDeterminFormerNum:   %lu", tailDeterminFormerNum_);
        OP_LOGD(tilingContext_->GetNodeName(), "tailDeterminTailNum:     %lu", tailDeterminTailNum_);

        OP_LOGD(tilingContext_->GetNodeName(), "partNum:                 %lu", partNum_);
        OP_LOGD(tilingContext_->GetNodeName(), "formerCopyRow:           %lu", formerCopyRow_);
        OP_LOGD(tilingContext_->GetNodeName(), "tailCopyRow:             %lu", tailCopyRow_);
        OP_LOGD(tilingContext_->GetNodeName(), "formerCopyTime:          %lu", formerCopyTime_);
        OP_LOGD(tilingContext_->GetNodeName(), "tailCopyTime:            %lu", tailCopyTime_);
        OP_LOGD(tilingContext_->GetNodeName(), "maxRowInUb:              %lu", maxRowInUb_);
        OP_LOGD(tilingContext_->GetNodeName(), "formerLastRow:           %lu", formerLastRow_);
        OP_LOGD(tilingContext_->GetNodeName(), "tailLastRow:             %lu", tailLastRow_);
    }

    ge::graphStatus TilingEmbeddingDenseGradV2(gert::TilingContext* context)
    {
        EmbeddingDenseGradV2Tiling tilingObject(context);
        if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
            OP_LOGE(context->GetNodeName(), "tiling init fail");
            return ge::GRAPH_FAILED;
        }
        return tilingObject.SetKernelTiling();
    }

    IMPL_OP_OPTILING(EmbeddingDenseGradV2)
        .Tiling(TilingEmbeddingDenseGradV2);
} // namespace optiling

namespace {
constexpr size_t EMBEDDING_DENSE_GRAD_IN_GRAD = 0;
constexpr size_t EMBEDDING_DENSE_GRAD_IN_INDICES = 1;
constexpr size_t EMBEDDING_DENSE_GRAD_IN_POSIDX = 2;
constexpr size_t EMBEDDING_DENSE_GRAD_OUT_Y = 0;
constexpr size_t EMBEDDING_DENSE_GRAD_ATTR_NUM_WEIGHTS = 0;
constexpr size_t EMBEDDING_DENSE_GRAD_ATTR_PADDING_IDX = 1;
constexpr size_t EMBEDDING_DENSE_GRAD_ATTR_SCALE_GRAD_BY_FREQ = 2;
}  // namespace

namespace ops {
static ge::graphStatus InferShapeForEmbeddingDenseGradV2(gert::InferShapeContext* context) {
  auto grad_shape = context->GetInputShape(EMBEDDING_DENSE_GRAD_IN_GRAD);
  OPS_CHECK_NULL_WITH_CONTEXT(context, grad_shape);
  auto indices_shape = context->GetInputShape(EMBEDDING_DENSE_GRAD_IN_INDICES);
  OPS_CHECK_NULL_WITH_CONTEXT(context, indices_shape);
  auto pos_shape = context->GetInputShape(EMBEDDING_DENSE_GRAD_IN_POSIDX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, pos_shape);
  auto out_shape = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, out_shape);
  auto attrs = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

  auto num_weights = attrs->GetAttrPointer<int64_t>(EMBEDDING_DENSE_GRAD_ATTR_NUM_WEIGHTS);
  OPS_CHECK_NULL_WITH_CONTEXT(context, num_weights);
  auto padding_idx = attrs->GetAttrPointer<int64_t>(EMBEDDING_DENSE_GRAD_ATTR_PADDING_IDX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, padding_idx);
  auto scale_grad_by_freq = attrs->GetAttrPointer<bool>(EMBEDDING_DENSE_GRAD_ATTR_SCALE_GRAD_BY_FREQ);
  OPS_CHECK_NULL_WITH_CONTEXT(context, scale_grad_by_freq);

  int64_t output_shape_len = 2;
  int64_t input_shape_len = grad_shape->GetDimNum();
  out_shape->SetDimNum(output_shape_len);
  out_shape->SetDim(0, *num_weights);
  out_shape->SetDim(1, grad_shape->GetDim(input_shape_len - 1));

  return GRAPH_SUCCESS;
}

static graphStatus InferDtypeForEmbeddingDenseGradV2(gert::InferDataTypeContext* context) {
  OP_LOGD(context->GetNodeName(), "Begin to do InferDtypeForEmbeddingDenseGrad rt2.0");
  auto grad_dtype = context->GetInputDataType(EMBEDDING_DENSE_GRAD_IN_GRAD);
  context->SetOutputDataType(EMBEDDING_DENSE_GRAD_OUT_Y, grad_dtype);
  OP_LOGD(context->GetNodeName(), "End to do InferDtypeForEmbeddingDenseGrad");

  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(EmbeddingDenseGradV2).InferShape(InferShapeForEmbeddingDenseGradV2)
                              .InferDataType(InferDtypeForEmbeddingDenseGradV2);
}  // namespace ops