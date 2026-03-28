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
 * \file reduce_sum_v2_tiling.cc
 * \brief
 */

#include "reduce_sum_v2_tiling.h"
#include "reduce_sum_v2_common.h"
#include "register/op_def_registry.h"
#include "platform/platform_info.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_base.h"
#include "tiling/tiling_type.h"

namespace {
namespace vectorutil {
int64_t CeilDiv(const int64_t dividend, const int64_t divisor) {
  if (divisor == 0) {
    return 0;
  }
  return (dividend + divisor - 1) / divisor;
}

int64_t FloorAlign(const int64_t dividend, const int64_t divisor) {
  if (divisor == 0) {
    return 0;
  }
  return dividend / divisor * divisor;
}
int64_t CeilAlign(const int64_t dividend, const int64_t divisor) {
  return CeilDiv(dividend, divisor) * divisor;
}
} // namespace vectorutil
}

namespace optiling {
    // Common
    constexpr uint64_t RESERVED_UB_SIZE = 10 * 2048;
    constexpr uint64_t CACHELINE = 512;
    constexpr uint64_t BLOCK_SIZE = 32;
    constexpr uint64_t BUFFER_NUM = 2;  // 开启DoubleBuffer
    constexpr uint64_t ROWS_IN_UB = 128;
    constexpr uint64_t UB_PROCESS_SIZE = 64 * 1024;
    constexpr uint64_t UB_CACHE_PROCESS_SIZE = 64 * 1024;
    constexpr uint64_t NUM_2 = 2;
    constexpr uint64_t NUM_3 = 3;

    // Tensor IDX
    constexpr uint64_t X_IDX = 0;
    constexpr uint64_t AXES_IDX = 1;
    // Pattern
    constexpr uint64_t PATTERN_AR = 0;
    constexpr uint64_t PATTERN_ARA = 1;

    struct CoreAndDataInfos {
        uint64_t formerCoreNum = 0;
        uint64_t tailCoreNum = 0;
        uint64_t formerUnitDataLen = 0;
        uint64_t tailUnitDataLen = 0;
        uint64_t tailRealDataLen = 0;
        uint64_t usedCoreNum = 1;
    };

    struct UbTilingInfos {
        uint64_t formerATimes = 0;
        uint64_t formerA = 0;
        uint64_t tailA = 0;
        uint64_t formerRTimes = 0;
        uint64_t formerR = 0;
        uint64_t tailR = 0;
        uint64_t formerA1Times = 0;
        uint64_t formerA1 = 0;
        uint64_t tailA1 = 0;
        uint64_t formerRealTimes = 0;
        uint64_t tailRealData = 0;
    };

    struct BlockTilingInfos {
        CoreAndDataInfos infosA;
        CoreAndDataInfos infosR;
        CoreAndDataInfos infosA1;
    };

    struct TilingInfos {
        uint64_t A = 1;
        uint64_t R = 1;
        uint64_t A1 = 1;
        uint64_t pattern = PATTERN_AR;
        uint64_t usedCoreNum = 1;
        uint64_t inBufferSize = 0;
        uint64_t cacheBufferSize = 0;
        uint64_t workspaceSize = 0;
        UbTilingInfos ubInfos[NUM_2];
        BlockTilingInfos blockInfos;
    };

    class ReduceSumV2Tiling {
    public:
        explicit ReduceSumV2Tiling(gert::TilingContext* context) : tilingContext_(context) {}
        ge::graphStatus Init();
        ge::graphStatus SetKernelTiling();
        void TilingDataPrint() const;
    private:
        inline void SetProcessTiling(ReduceSumV2Process &tiling, const uint64_t &processId);
        inline void SetTilingKeyMode();
        inline void PrintBlockInfos(const BlockTilingInfos &blockInfos) const;
        inline void PrintUbInfos(const UbTilingInfos &ubInfos) const;
        inline uint64_t CalcDataSize(const uint64_t &A, const uint64_t &R);
        inline void Tiling4Base();
        inline void UbTiling(BlockTilingInfos &blockInfos, const uint64_t &pattern, UbTilingInfos (&ubInfos)[NUM_2]);
        inline void UbTilingAR(const uint64_t &cutA, const uint64_t &cutR, const uint64_t &tailRealData, UbTilingInfos &ubInfos);
        inline void UbTilingARA(const uint64_t &cutA, const uint64_t &cutR, const uint64_t &cutA1, const uint64_t &tailRealData, UbTilingInfos &ubInfos);
        inline void BlockTiling(TilingInfos &tilingInfos);
        inline void BlockTilingAR(TilingInfos &tilingInfos);
        inline void BlockTilingARA(TilingInfos &tilingInfos);
        inline void CalcCoreAndDataLen(uint64_t coreNum, const uint64_t &dataLen, const uint64_t &unitDataLen, CoreAndDataInfos &coreDataInfos);
        inline uint64_t CalcTilingKey();
        inline void CalcUbAndWsSize(TilingInfos &tilingInfos, const uint64_t &preocessId);

        ReduceSumV2TilingData tilingData_;
        gert::TilingContext* tilingContext_ = nullptr;
    private:
        uint64_t tilingKey_ = 0;
        uint64_t coreNum_ = 0;
        uint64_t usedCoreNum_ = 0;
        uint64_t ubSize_ = 0;
        uint64_t availableUb_ = 0;
        std::vector<uint64_t> xShape_;
        std::vector<uint64_t> axes_;
        ge::DataType xDtype_;
        uint64_t xDtypeSize_ = 4;
        bool keepDims_ = false;
        uint64_t processNum_ = 0;       // process次数，范围[1, 4]
        uint64_t usedWorkspaceSize_ = 0;

        TilingInfos tilingInfosList[4];
    };

    inline void ReduceSumV2Tiling::SetTilingKeyMode()
    {
        tilingKey_ = CalcTilingKey();
        tilingContext_->SetTilingKey(tilingKey_);
    }

    ge::graphStatus ReduceSumV2Tiling::Init()
    {
        OP_LOGD(tilingContext_->GetNodeName(), "ReduceSumV2Tiling initing");
        auto compileInfo = reinterpret_cast<const ReduceSumV2CompileInfo*>(tilingContext_->GetCompileInfo());
        auto attrs = tilingContext_->GetAttrs();
        keepDims_ = *(attrs->GetAttrPointer<uint64_t>)(0);
        std::cout << "keepDims_: " << keepDims_ << std::endl;
        auto xShape = tilingContext_->GetInputShape(X_IDX)->GetStorageShape();
        for (size_t i = 0; i < xShape.GetDimNum(); i++) {
            xShape_.emplace_back(xShape.GetDim(i));
        }

        ge::DataType axesDtype;
        ge::graphStatus status = ReduceCommon::GetInputDtype(tilingContext_, X_IDX, xDtype_);
        OP_TILING_CHECK((status == ge::GRAPH_FAILED),
                        VECTOR_INNER_ERR_REPORT_TILIING(tilingContext_->GetNodeName(), "ReduceSum get axes dtype failed"),
                        return ge::GRAPH_FAILED);
        status = ReduceCommon::GetInputDtype(tilingContext_, AXES_IDX, axesDtype);
        OP_TILING_CHECK((status == ge::GRAPH_FAILED),
                        VECTOR_INNER_ERR_REPORT_TILIING(tilingContext_->GetNodeName(), "ReduceSum get axes dtype failed"),
                        return ge::GRAPH_FAILED);
        if (axesDtype == ge::DT_INT32) {
            status = ReduceCommon::GetAxesData<int32_t>(tilingContext_, AXES_IDX, axes_);
        } else if (axesDtype == ge::DT_INT64) {
            status = ReduceCommon::GetAxesData<int64_t>(tilingContext_, AXES_IDX, axes_);
        }
        OP_TILING_CHECK((status == ge::GRAPH_FAILED),
                        VECTOR_INNER_ERR_REPORT_TILIING(tilingContext_->GetNodeName(), "ReduceSum get axes const input failed"),
                        return ge::GRAPH_FAILED);
        
        xDtypeSize_ = sizeof(float);
        
        coreNum_ = compileInfo->totalCoreNum;      
        ubSize_ = compileInfo->ubSizePlatForm;
        availableUb_ = std::min(ubSize_ - RESERVED_UB_SIZE, UB_PROCESS_SIZE);

        Tiling4Base();
        
        size_t sysWorkspaceSize = compileInfo->sysWorkspaceSize;
        sysWorkspaceSize = sysWorkspaceSize + usedWorkspaceSize_;
        size_t *currentWorkSpace = tilingContext_->GetWorkspaceSizes(1);
        currentWorkSpace[0] = sysWorkspaceSize;
        SetTilingKeyMode();
        
        OP_LOGD(tilingContext_->GetNodeName(), "ReduceSumV2Tiling inited");
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus ReduceSumV2Tiling::SetKernelTiling()
    {
        for (auto tilingInfos : tilingInfosList) {
            usedCoreNum_ = usedCoreNum_ < tilingInfos.usedCoreNum ? tilingInfos.usedCoreNum : usedCoreNum_;
        }
        tilingContext_->SetBlockDim(usedCoreNum_);
        tilingData_.set_processNum(processNum_);
        SetProcessTiling(tilingData_.process0, 0);
        SetProcessTiling(tilingData_.process1, 1);
        SetProcessTiling(tilingData_.process2, NUM_2);
        SetProcessTiling(tilingData_.process3, NUM_3);

        tilingData_.SaveToBuffer(tilingContext_->GetRawTilingData()->GetData(),
                                tilingContext_->GetRawTilingData()->GetCapacity());
        tilingContext_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
        TilingDataPrint();
        return ge::GRAPH_SUCCESS;
    }

    inline void ReduceSumV2Tiling::SetProcessTiling(ReduceSumV2Process &tiling, const uint64_t &processId)
    {
        tiling.set_A(tilingInfosList[processId].A);
        tiling.set_R(tilingInfosList[processId].R);
        tiling.set_A1(tilingInfosList[processId].A1);
        tiling.set_pattern(tilingInfosList[processId].pattern);
        tiling.set_usedCoreNum(tilingInfosList[processId].usedCoreNum);
        tiling.set_inBufferSize(UB_PROCESS_SIZE / BUFFER_NUM);
        tiling.set_cacheBufferSize(UB_CACHE_PROCESS_SIZE);

        size_t ubInfosNum = NUM_2;
        
        const UbTilingInfos *ubInfos = tilingInfosList[processId].ubInfos;
        uint64_t formerATimes[ubInfosNum];
        uint64_t formerA[ubInfosNum];
        uint64_t tailA[ubInfosNum];
        uint64_t formerRTimes[ubInfosNum];
        uint64_t formerR[ubInfosNum];
        uint64_t tailR[ubInfosNum];
        uint64_t formerA1Times[ubInfosNum];
        uint64_t formerA1[ubInfosNum];
        uint64_t tailA1[ubInfosNum];
        uint64_t formerRealTimes[ubInfosNum];
        uint64_t tailRealData[ubInfosNum];

        for (size_t j = 0; j < ubInfosNum; j++) {
            formerATimes[j] = ubInfos[j].formerATimes;
            formerA[j] = ubInfos[j].formerA;
            tailA[j] = ubInfos[j].tailA;
            formerRTimes[j] = ubInfos[j].formerRTimes;
            formerR[j] = ubInfos[j].formerR;
            tailR[j] = ubInfos[j].tailR;
            formerA1Times[j] = ubInfos[j].formerA1Times;
            formerA1[j] = ubInfos[j].formerA1;
            tailA1[j] = ubInfos[j].tailA1;
            formerRealTimes[j] = ubInfos[j].formerRealTimes; 
            tailRealData[j] = ubInfos[j].tailRealData; 
        }
        tiling.ubInfos.set_formerATimes(formerATimes);
        tiling.ubInfos.set_formerA(formerA);
        tiling.ubInfos.set_tailA(tailA);
        tiling.ubInfos.set_formerRTimes(formerRTimes);
        tiling.ubInfos.set_formerR(formerR);
        tiling.ubInfos.set_tailR(tailR);
        tiling.ubInfos.set_formerA1Times(formerA1Times);
        tiling.ubInfos.set_formerA1(formerA1);
        tiling.ubInfos.set_tailA1(tailA1);
        tiling.ubInfos.set_formerRealTimes(formerRealTimes);
        tiling.ubInfos.set_tailRealData(tailRealData);

        CoreAndDataInfos &blockInfosA = tilingInfosList[processId].blockInfos.infosA;
        tiling.blockA.set_usedCoreNum(blockInfosA.usedCoreNum);
        tiling.blockA.set_formerCoreNum(blockInfosA.formerCoreNum);
        tiling.blockA.set_formerUnitDataLen(blockInfosA.formerUnitDataLen);
        tiling.blockA.set_tailUnitDataLen(blockInfosA.tailUnitDataLen);
        tiling.blockA.set_tailRealDataLen(blockInfosA.tailRealDataLen);

        CoreAndDataInfos &blockInfosR = tilingInfosList[processId].blockInfos.infosR;
        tiling.blockR.set_usedCoreNum(blockInfosR.usedCoreNum);
        tiling.blockR.set_formerCoreNum(blockInfosR.formerCoreNum);
        tiling.blockR.set_formerUnitDataLen(blockInfosR.formerUnitDataLen);
        tiling.blockR.set_tailUnitDataLen(blockInfosR.tailUnitDataLen);
        tiling.blockR.set_tailRealDataLen(blockInfosR.tailRealDataLen);

        CoreAndDataInfos &blockInfosA1 = tilingInfosList[processId].blockInfos.infosA1;
        tiling.blockA1.set_usedCoreNum(blockInfosA1.usedCoreNum);
        tiling.blockA1.set_formerCoreNum(blockInfosA1.formerCoreNum);
        tiling.blockA1.set_formerUnitDataLen(blockInfosA1.formerUnitDataLen);
        tiling.blockA1.set_tailUnitDataLen(blockInfosA1.tailUnitDataLen);
        tiling.blockA1.set_tailRealDataLen(blockInfosA1.tailRealDataLen);
    }

    void ReduceSumV2Tiling::TilingDataPrint() const
    {
        OP_LOGD(tilingContext_->GetNodeName(), "tilingKey:                  %lu", tilingKey_);
        OP_LOGD(tilingContext_->GetNodeName(), "coreNum:                    %lu", coreNum_);
        OP_LOGD(tilingContext_->GetNodeName(), "ubSize:                     %lu", ubSize_);
        OP_LOGD(tilingContext_->GetNodeName(), "usedCoreNum:                %lu", usedCoreNum_);
        OP_LOGD(tilingContext_->GetNodeName(), "processNum:                 %lu", processNum_);
        for (size_t i = 0; i < processNum_; i++) {
            OP_LOGD(tilingContext_->GetNodeName(), "--- process %lu tiling data ---", i);
            OP_LOGD(tilingContext_->GetNodeName(), "A:                          %lu", tilingInfosList[i].A);
            OP_LOGD(tilingContext_->GetNodeName(), "R:                          %lu", tilingInfosList[i].R);
            OP_LOGD(tilingContext_->GetNodeName(), "A1:                         %lu", tilingInfosList[i].A1);
            OP_LOGD(tilingContext_->GetNodeName(), "pattern:                    %lu", tilingInfosList[i].pattern);
            OP_LOGD(tilingContext_->GetNodeName(), "usedCoreNum:                %lu", tilingInfosList[i].usedCoreNum);
            OP_LOGD(tilingContext_->GetNodeName(), "inBufferSize:               %lu", tilingInfosList[i].inBufferSize);
            OP_LOGD(tilingContext_->GetNodeName(), "cacheBufferSize:            %lu", tilingInfosList[i].cacheBufferSize);
            OP_LOGD(tilingContext_->GetNodeName(), "workspaceSize:              %lu", tilingInfosList[i].workspaceSize);
            PrintBlockInfos(tilingInfosList[i].blockInfos);
            for (size_t j = 0; j < NUM_2; j++) {
                OP_LOGD(tilingContext_->GetNodeName(), "--- ubInfos %lu tiling data ---", i);
                PrintUbInfos(tilingInfosList[i].ubInfos[j]);
            }
        }  
    }

    inline void ReduceSumV2Tiling::PrintBlockInfos(const BlockTilingInfos &blockInfos) const
    {
        OP_LOGD(tilingContext_->GetNodeName(), "[A]:usedCoreNum:            %lu", blockInfos.infosA.usedCoreNum);
        OP_LOGD(tilingContext_->GetNodeName(), "[A]:formerCoreNum:          %lu", blockInfos.infosA.formerCoreNum);
        OP_LOGD(tilingContext_->GetNodeName(), "[A]:tailCoreNum:            %lu", blockInfos.infosA.tailCoreNum);
        OP_LOGD(tilingContext_->GetNodeName(), "[A]:formerUnitDataLen:      %lu", blockInfos.infosA.formerUnitDataLen);
        OP_LOGD(tilingContext_->GetNodeName(), "[A]:tailUnitDataLen:        %lu", blockInfos.infosA.tailUnitDataLen);
        OP_LOGD(tilingContext_->GetNodeName(), "[A]:tailRealDataLen:        %lu", blockInfos.infosA.tailRealDataLen);
        
        OP_LOGD(tilingContext_->GetNodeName(), "[R]:usedCoreNum:            %lu", blockInfos.infosR.usedCoreNum);
        OP_LOGD(tilingContext_->GetNodeName(), "[R]:formerCoreNum:          %lu", blockInfos.infosR.formerCoreNum);
        OP_LOGD(tilingContext_->GetNodeName(), "[R]:tailCoreNum:            %lu", blockInfos.infosR.tailCoreNum);
        OP_LOGD(tilingContext_->GetNodeName(), "[R]:formerUnitDataLen:      %lu", blockInfos.infosR.formerUnitDataLen);
        OP_LOGD(tilingContext_->GetNodeName(), "[R]:tailUnitDataLen:        %lu", blockInfos.infosR.tailUnitDataLen);
        OP_LOGD(tilingContext_->GetNodeName(), "[R]:tailRealDataLen:        %lu", blockInfos.infosR.tailRealDataLen);

        OP_LOGD(tilingContext_->GetNodeName(), "[A1]:usedCoreNum:           %lu", blockInfos.infosA1.usedCoreNum);
        OP_LOGD(tilingContext_->GetNodeName(), "[A1]:formerCoreNum:         %lu", blockInfos.infosA1.formerCoreNum);
        OP_LOGD(tilingContext_->GetNodeName(), "[A1]:tailCoreNum:           %lu", blockInfos.infosA1.tailCoreNum);
        OP_LOGD(tilingContext_->GetNodeName(), "[A1]:formerUnitDataLen:     %lu", blockInfos.infosA1.formerUnitDataLen);
        OP_LOGD(tilingContext_->GetNodeName(), "[A1]:tailUnitDataLen:       %lu", blockInfos.infosA1.tailUnitDataLen);
        OP_LOGD(tilingContext_->GetNodeName(), "[A1]:tailRealDataLen:       %lu", blockInfos.infosA1.tailRealDataLen);
    }

    inline void ReduceSumV2Tiling::PrintUbInfos(const UbTilingInfos &ubInfos) const
    {
        OP_LOGD(tilingContext_->GetNodeName(), "formerATimes                %lu", ubInfos.formerATimes);
        OP_LOGD(tilingContext_->GetNodeName(), "formerA:                    %lu", ubInfos.formerA);
        OP_LOGD(tilingContext_->GetNodeName(), "tailA:                      %lu", ubInfos.tailA);
        OP_LOGD(tilingContext_->GetNodeName(), "formerRTimes                %lu", ubInfos.formerRTimes);
        OP_LOGD(tilingContext_->GetNodeName(), "formerR:                    %lu", ubInfos.formerR);
        OP_LOGD(tilingContext_->GetNodeName(), "tailR:                      %lu", ubInfos.tailR);
        OP_LOGD(tilingContext_->GetNodeName(), "formerA1Times               %lu", ubInfos.formerA1Times);
        OP_LOGD(tilingContext_->GetNodeName(), "formerA1:                   %lu", ubInfos.formerA1);
        OP_LOGD(tilingContext_->GetNodeName(), "tailA1:                     %lu", ubInfos.tailA1);
        OP_LOGD(tilingContext_->GetNodeName(), "formerRealTimes:            %lu", ubInfos.formerRealTimes);
        OP_LOGD(tilingContext_->GetNodeName(), "tailRealData:               %lu", ubInfos.tailRealData);
    }

    ge::graphStatus Tiling4ReduceSumV2(gert::TilingContext* context)
    {
        OP_LOGD(context->GetNodeName(), "TilingReduceSumV2 enter.");
        if (context == nullptr) {
            OP_LOGE(context->GetNodeName(), "The context is nullptr.");
            return ge::GRAPH_FAILED;
        }

        ReduceSumV2Tiling tilingObject(context);
        if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
            OP_LOGE(context->GetNodeName(), "tiling init fail");
            return ge::GRAPH_FAILED;
        }
        return tilingObject.SetKernelTiling();
    }

    ge::graphStatus TilingPrepare4ReduceSumV2(gert::TilingParseContext* context)
    {
        OP_LOGD(context->GetNodeName(), "Tiling Prepare For ReduceSumV2 start");
        auto compileInfo = GetCompileInfoPtr<ReduceSumV2CompileInfo>(context);
        OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
        auto platformInfo = context->GetPlatformInfo();
        OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
        OP_TILING_CHECK((compileInfo->ubSizePlatForm <= 0),
                        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Failed to get ub size"),
                        return ge::GRAPH_FAILED);
        OP_LOGD(context->GetNodeName(), "ub_size_platform is %lu", compileInfo->ubSizePlatForm);
        uint64_t totalUbSize = 0;
        platformInfo->GetLocalMemSize(fe::LocalMemType::UB, totalUbSize);
        OP_LOGD(context->GetNodeName(), "total ub size is %lu", totalUbSize);
        compileInfo->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
        OP_TILING_CHECK((compileInfo->sysWorkspaceSize < 0),
                        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                        "sysWorkspaceSize should be greater than or equal to zero"),
                        return ge::GRAPH_FAILED);
        OP_LOGD(context->GetNodeName(), "Tiling prepare for ReduceSumV2 end");
        return ge::GRAPH_SUCCESS;
    }
    
    inline uint64_t ReduceSumV2Tiling::CalcDataSize(const uint64_t &A, const uint64_t &R)
    {
        return A * vectorutil::CeilAlign(R * xDtypeSize_, BLOCK_SIZE);
    }

    inline void ReduceSumV2Tiling::Tiling4Base()
    {
        OP_LOGD(tilingContext_->GetNodeName(), "Tiling4Base start");
        std::vector<uint64_t> xFusedShape;
        ReduceCommon::DoFuseAxes(xShape_, axes_, xFusedShape, processNum_);
        for (size_t processId = 0; processId < processNum_; processId++) {
            // shape推导
            tilingInfosList[processId].pattern = processId == processNum_ - 1 ? xFusedShape.size() % 2 : PATTERN_ARA;
            uint64_t lastA = processId ? tilingInfosList[processId - 1].A : 1;
            tilingInfosList[processId].A = lastA * xFusedShape[processId * 2];
            tilingInfosList[processId].R = xFusedShape[processId * 2 + 1];
            tilingInfosList[processId].A1 = tilingInfosList[processId].pattern == PATTERN_AR ? 1 :
                                            std::accumulate(xFusedShape.begin() + (processId + 1) * 2, xFusedShape.end(), 1LL, std::multiplies<uint64_t>());
            BlockTiling(tilingInfosList[processId]);
            UbTiling(tilingInfosList[processId].blockInfos, tilingInfosList[processId].pattern, tilingInfosList[processId].ubInfos);
            CalcUbAndWsSize(tilingInfosList[processId], processId);
        }

        OP_LOGD(tilingContext_->GetNodeName(), "Tiling4Base finished");
    }

    // UBTiling
    inline void ReduceSumV2Tiling::UbTiling(BlockTilingInfos &blockInfos, const uint64_t &pattern, UbTilingInfos (&ubInfos)[NUM_2])
    {
        if (pattern == PATTERN_AR) {
            if (blockInfos.infosR.usedCoreNum != 1) {
                UbTilingAR(blockInfos.infosA.tailUnitDataLen, blockInfos.infosR.formerUnitDataLen, blockInfos.infosR.formerUnitDataLen, ubInfos[0]);
            } else {
                UbTilingAR(blockInfos.infosA.formerUnitDataLen, blockInfos.infosR.tailUnitDataLen, blockInfos.infosR.tailRealDataLen, ubInfos[0]);
            }
            UbTilingAR(blockInfos.infosA.tailUnitDataLen, blockInfos.infosR.tailUnitDataLen, blockInfos.infosR.tailRealDataLen, ubInfos[1]);
        } else if (pattern == PATTERN_ARA) {
            if (blockInfos.infosR.usedCoreNum != 1) {
                UbTilingARA(blockInfos.infosA.tailUnitDataLen, blockInfos.infosR.formerUnitDataLen, blockInfos.infosA1.tailUnitDataLen, blockInfos.infosA1.tailRealDataLen, ubInfos[0]);
            } else if (blockInfos.infosA1.usedCoreNum != 1) {
                UbTilingARA(blockInfos.infosA.tailUnitDataLen, blockInfos.infosR.tailUnitDataLen, blockInfos.infosA1.formerUnitDataLen, blockInfos.infosA1.formerUnitDataLen, ubInfos[0]);
            } else {
                UbTilingARA(blockInfos.infosA.formerUnitDataLen, blockInfos.infosR.tailUnitDataLen, blockInfos.infosA1.tailUnitDataLen, blockInfos.infosA1.tailRealDataLen, ubInfos[0]);
            }
            UbTilingARA(blockInfos.infosA.tailUnitDataLen, blockInfos.infosR.tailUnitDataLen, blockInfos.infosA1.tailUnitDataLen, blockInfos.infosA1.tailRealDataLen, ubInfos[1]);
        } else {
            std::cout << "pattern do not support yet" << std::endl;
        }
    }

    inline void ReduceSumV2Tiling::UbTilingAR(const uint64_t &cutA, const uint64_t &cutR, const uint64_t &tailRealData, UbTilingInfos &ubInfos)
    {
        // 当前切块是否能全载
        uint64_t dataAlignSize = CalcDataSize(cutA, cutR);
        uint64_t eachBufferSize = availableUb_ / BUFFER_NUM;
        // UB不切分
        ubInfos.formerATimes = 0;
        ubInfos.formerA = 0;
        ubInfos.tailA = cutA;
        ubInfos.formerRTimes = 0;
        ubInfos.formerR = 0;
        ubInfos.tailR = cutR;
        ubInfos.formerRealTimes = 0;
        ubInfos.tailRealData = tailRealData;
        if (dataAlignSize <= eachBufferSize) {
            return;
        }
        // UB切A
        dataAlignSize = CalcDataSize(1, cutR);
        if (dataAlignSize <= eachBufferSize) {
            ubInfos.formerA = eachBufferSize / vectorutil::CeilAlign(cutR * xDtypeSize_, BLOCK_SIZE);
            ubInfos.formerATimes = cutA / ubInfos.formerA;
            ubInfos.tailA = cutA % ubInfos.formerA;
            return;
        }
        // UB切AR
        ubInfos.formerA = 1;
        ubInfos.formerATimes = cutA / ubInfos.formerA;
        ubInfos.tailA = cutA % ubInfos.formerA;
        if (cutA == 1) {
            ubInfos.formerA = 0;
            ubInfos.formerATimes = 0;
            ubInfos.tailA = 1;
        }
        ubInfos.formerR = vectorutil::FloorAlign(eachBufferSize, CACHELINE) / xDtypeSize_; // 每行多少个CACHELINE
        ubInfos.formerRTimes = cutR / ubInfos.formerR;
        ubInfos.tailR = cutR % ubInfos.formerR;

        ubInfos.formerRealTimes = tailRealData / ubInfos.formerR;
        ubInfos.tailRealData = tailRealData % ubInfos.formerR;
    }

    inline void ReduceSumV2Tiling::UbTilingARA(const uint64_t &cutA, const uint64_t &cutR, const uint64_t &cutA1, const uint64_t &tailRealData, UbTilingInfos &ubInfos)
    {
        // 当前切块是否能全载
        uint64_t dataAlignSize = CalcDataSize(cutA * cutR, cutA1);
        uint64_t eachBufferSize = availableUb_ / BUFFER_NUM;
        // UB不切分
        ubInfos.formerATimes = 0;
        ubInfos.formerA = 0;
        ubInfos.tailA = cutA;
        ubInfos.formerRTimes = 0;
        ubInfos.formerR = 0;
        ubInfos.tailR = cutR;
        ubInfos.formerA1Times = 0;
        ubInfos.formerA1 = 0;
        ubInfos.tailA1 = cutA1;
        ubInfos.formerRealTimes = 0;
        ubInfos.tailRealData = tailRealData;
        if (dataAlignSize <= eachBufferSize) {
            return;
        }

        // UB切A，A最小单位为1
        dataAlignSize = CalcDataSize(1 * cutR, cutA1);
        if (dataAlignSize <= eachBufferSize) {
            ubInfos.formerA = eachBufferSize / vectorutil::CeilAlign(cutR * cutA1 * xDtypeSize_, BLOCK_SIZE);
            ubInfos.formerA = eachBufferSize / (cutR * cutA1);
            ubInfos.formerATimes = cutA / ubInfos.formerA;
            ubInfos.tailA = cutA % ubInfos.formerA;
            return;
        }
        // UB切A1，A1最小单位为CACHELINE
        ubInfos.formerA = 1;
        ubInfos.formerATimes = cutA / ubInfos.formerA;
        ubInfos.tailA = cutA % ubInfos.formerA;
        if (ubInfos.formerATimes == 1) {
            ubInfos.formerA = 0;
            ubInfos.formerATimes = 0;
            ubInfos.tailA = 1;
        }
        dataAlignSize = CalcDataSize(1 * cutR, CACHELINE / xDtypeSize_);
        if (dataAlignSize <= eachBufferSize) {
            ubInfos.formerA1 = eachBufferSize / (cutR * CACHELINE) * (CACHELINE / xDtypeSize_); // 每行多少个CACHELINE
            ubInfos.formerA1Times = cutA1 / ubInfos.formerA1;
            ubInfos.tailA1 = cutA1 % ubInfos.formerA1;
            // 最后一个尾块
            ubInfos.formerRealTimes = tailRealData / ubInfos.formerA1;
            ubInfos.tailRealData = tailRealData % ubInfos.formerA1;
            return;
        }
        // UB切R，R最小单位为2
        ubInfos.formerA1 = CACHELINE / xDtypeSize_;
        ubInfos.formerA1Times = cutA1 / ubInfos.formerA1;
        ubInfos.tailA1 = cutA1 % ubInfos.formerA1;

        ubInfos.formerRealTimes = tailRealData / ubInfos.formerA1;
        ubInfos.tailRealData = tailRealData % ubInfos.formerA1;
        if (ubInfos.formerA1Times == 1) {
            ubInfos.tailA1 = ubInfos.formerA1;
            ubInfos.tailRealData = tailRealData;
            ubInfos.formerA1 = 0;
            ubInfos.formerA1Times = 0;
            ubInfos.formerRealTimes = 0;
        }
        ubInfos.formerR = eachBufferSize / CACHELINE;
        ubInfos.formerRTimes = cutR / ubInfos.formerR;
        ubInfos.tailR = cutR % ubInfos.formerR;
        return;
    }

    // BlockTiling
    inline void ReduceSumV2Tiling::BlockTiling(TilingInfos &tilingInfos)
    {
        if (tilingInfos.pattern == PATTERN_AR) {
            BlockTilingAR(tilingInfos);
        } else if (tilingInfos.pattern == PATTERN_ARA) {
            BlockTilingARA(tilingInfos);
        } else {
            std::cout << "pattern do not support yet" << std::endl;
        }
    }

    inline void ReduceSumV2Tiling::BlockTilingAR(TilingInfos &tilingInfos)
    {
        // A开核
        CalcCoreAndDataLen(coreNum_, tilingInfos.A, 1, tilingInfos.blockInfos.infosA);
        uint64_t remainCore4R = coreNum_ / tilingInfos.blockInfos.infosA.usedCoreNum;
        // AR一起开核
        CalcCoreAndDataLen(remainCore4R, tilingInfos.R, CACHELINE / xDtypeSize_, tilingInfos.blockInfos.infosR);        
        tilingInfos.usedCoreNum = tilingInfos.blockInfos.infosA.usedCoreNum * tilingInfos.blockInfos.infosR.usedCoreNum;
    }

    inline void ReduceSumV2Tiling::BlockTilingARA(TilingInfos &tilingInfos)
    {
        // A先开多核
        CalcCoreAndDataLen(coreNum_, tilingInfos.A, 1, tilingInfos.blockInfos.infosA);
        uint64_t remainCore4A1 = coreNum_ / tilingInfos.blockInfos.infosA.usedCoreNum;
        CalcCoreAndDataLen(remainCore4A1, tilingInfos.A1, CACHELINE / xDtypeSize_, tilingInfos.blockInfos.infosA1);
        uint64_t remainCore4R = remainCore4A1 / tilingInfos.blockInfos.infosA1.usedCoreNum;
        // 最少每个核处理两个R
        remainCore4R = std::min(std::max(tilingInfos.R / 2, static_cast<uint64_t>(1)), remainCore4R);
        CalcCoreAndDataLen(remainCore4R, tilingInfos.R, 1, tilingInfos.blockInfos.infosR);
        tilingInfos.usedCoreNum = tilingInfos.blockInfos.infosA.usedCoreNum * tilingInfos.blockInfos.infosR.usedCoreNum * tilingInfos.blockInfos.infosA1.usedCoreNum;
    }

    inline void ReduceSumV2Tiling::CalcCoreAndDataLen(uint64_t coreNum, const uint64_t &dataLen, const uint64_t &unitDataLen, CoreAndDataInfos &coreDataInfos)
    {
        uint64_t unitDataNum = vectorutil::CeilDiv(dataLen, unitDataLen);
        coreNum = std::min(unitDataNum, coreNum); // 数据量少于核数的情况
        coreDataInfos.formerCoreNum = unitDataNum % std::max(static_cast<uint64_t>(1), coreNum);
        coreDataInfos.tailCoreNum = coreNum - coreDataInfos.formerCoreNum;
        coreDataInfos.formerUnitDataLen = vectorutil::CeilDiv(unitDataNum, coreNum) * unitDataLen;
        coreDataInfos.tailUnitDataLen = unitDataNum / std::max(static_cast<uint64_t>(1), coreNum) * unitDataLen;
        // 最后一个核实际处理数据
        coreDataInfos.tailRealDataLen = dataLen - (coreDataInfos.formerCoreNum * coreDataInfos.formerUnitDataLen + (coreDataInfos.tailCoreNum - 1) * coreDataInfos.tailUnitDataLen);
        coreDataInfos.usedCoreNum = coreDataInfos.formerCoreNum + coreDataInfos.tailCoreNum;
    }

    inline void ReduceSumV2Tiling::CalcUbAndWsSize(TilingInfos &tilingInfos, const uint64_t &preocessId)
    {
        uint64_t syncWorkspace = 0;
        uint64_t tempWorkspace = 0;
        if (tilingInfos.blockInfos.infosR.usedCoreNum != 1) {
            tempWorkspace = tilingInfos.usedCoreNum * CACHELINE;
            syncWorkspace = tilingInfos.blockInfos.infosA.usedCoreNum *
                            tilingInfos.blockInfos.infosR.usedCoreNum *
                            tilingInfos.blockInfos.infosA1.usedCoreNum * BLOCK_SIZE;
        }
        uint64_t outWorkspace = 0;
        if (preocessId < processNum_ - 1) {
            outWorkspace = tilingInfos.A * tilingInfos.A1 * xDtypeSize_;
        }
        tilingInfos.workspaceSize = syncWorkspace + tempWorkspace + outWorkspace;
        usedWorkspaceSize_ = std::max(usedWorkspaceSize_, tilingInfos.workspaceSize);
    }

    inline uint64_t ReduceSumV2Tiling::CalcTilingKey()
    {
        uint64_t tilingkey;
        switch (processNum_) {
            case 1:
                tilingkey = tilingInfosList[0].pattern;
                break;
            case 2:
                tilingkey = RecursiveSum(tilingInfosList[1].pattern, tilingInfosList[0].pattern);
                break;
            case 3:
                tilingkey = RecursiveSum(tilingInfosList[2].pattern, tilingInfosList[1].pattern,
                                         tilingInfosList[0].pattern);
                break;
            case 4:
                tilingkey = RecursiveSum(tilingInfosList[3].pattern, tilingInfosList[2].pattern,
                                         tilingInfosList[1].pattern, tilingInfosList[0].pattern);
                break;
            default:
                tilingkey = 0;
        }
        return tilingkey;
    }

    IMPL_OP_OPTILING(ReduceSumV2)
        .Tiling(Tiling4ReduceSumV2)
        .TilingParse<ReduceSumV2CompileInfo>(TilingPrepare4ReduceSumV2);

} // namespace optiling