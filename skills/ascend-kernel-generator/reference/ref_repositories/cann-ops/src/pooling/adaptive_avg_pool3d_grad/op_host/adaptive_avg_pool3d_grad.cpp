/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file adaptive_avg_pool3d_grad.cpp
 * \brief
 */

#include <iostream>
#include "tiling/tiling_api.h"
#include "tiling/tiling_type.h"
#include "adaptive_avg_pool3d_grad_tiling.h"
#include "register/op_def_registry.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"


namespace optiling {
constexpr int64_t D_DIM = 0;
    constexpr int64_t H_DIM = 1;
    constexpr int64_t W_DIM = 2;
    constexpr int64_t RESERVED_UB = 3 * 1024;
    constexpr int64_t INDEX_USE_UB = 3 * 1024;
    constexpr int64_t ALIGN_SIZE = 32;
    constexpr int64_t BF_FP16_DSIZE = 2;
    constexpr int64_t FP32_DSIZE = 4;

    constexpr int64_t Y_GRAD_DIMS = 4;
    constexpr int64_t X_DIMS_4 = 4;
    constexpr int64_t X_DIMS_5 = 5;

    constexpr int64_t FP32_DTYPE_KEY = 0;
    constexpr int64_t FP16_DTYPE_KEY = 1;
    constexpr int64_t BF16_DTYPE_KEY = 2;

    constexpr int64_t NC_SMALL_KEY = 0;
    constexpr int64_t NC_LARGE_KEY = 1;
    constexpr int64_t DTYPE_KEY_WEIGHT = 10;

    constexpr size_t DIM0 = 0;
    constexpr size_t DIM1 = 1;
    constexpr size_t DIM2 = 2;
    constexpr size_t DIM3 = 3;
    constexpr size_t DIM4 = 4;

    class AdaptiveAvgPool3dGradTiling{
    public:
        explicit AdaptiveAvgPool3dGradTiling(gert::TilingContext* context): context(context){};
        ge::graphStatus Init();
        ge::graphStatus RunKernelTiling();
        void CalTilingKey(uint32_t ubSize);
        bool GetDataTypeKey(ge::DataType dataType);
    private:
        AdaptiveAvgPool3dGradTilingData TilingData;
        gert::TilingContext* context=nullptr;
        int64_t sysWorkspaceSize = 16 * 1024 *1024;
        int64_t useWorkspaceSize = 0;
        int64_t blockBytes = 32;
        int64_t dataAlign = 4;
        int64_t yGradDSize = 4;
        uint32_t coreNum = 48;
        int64_t dOut = 1;
        int64_t hOut = 1;
        int64_t wOut = 1;
        int64_t dIn = 1;
        int64_t hIn = 1;
        int64_t wIn = 1;
        int64_t ncNum = 1;
        int64_t ncAlign = 1;
        int64_t yGradNum = 0;
        int64_t taskCoreUsed = 0;

        int64_t yNumPerCalc = 0;
        int64_t taskNum = 0;
        int64_t taskNumPerCore = 0;
        int64_t taskNumLastCore = 0;
        int64_t indexCalcCoreUsed = 0;
        int64_t indexNumPerCore = 0;
        int64_t indexNumLastCore = 0;
        int64_t kW = 0;
        int64_t kH = 0;
        int64_t kD = 0;
        int64_t tilingKey = 0;

        int64_t perCalcSize = 0;
        int64_t ncSliceNum = 0;
        int64_t ncAlignSliceLength = 0;
        int64_t ncAlignSliceTail = 0;

        int64_t isAtomicAdd = 1;
        int64_t deterministicFlag = 0;
    };

    ge::graphStatus AdaptiveAvgPool3dGradTiling::Init() {
        auto nodeName = context->GetNodeName();

        auto platformInfo = context->GetPlatformInfo();
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        uint64_t ubSizePlatForm;
    
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        uint32_t ubSize = static_cast<uint32_t>(ubSizePlatForm) - RESERVED_UB - INDEX_USE_UB; 
        sysWorkspaceSize =  ascendcPlatform.GetLibApiWorkSpaceSize();
        coreNum = ascendcPlatform.GetCoreNumAiv();
    
        deterministicFlag = context->GetDeterministic() == 1 ? 1 : 0;
        if (deterministicFlag == 1) {
            coreNum = 1;
        }
        auto const yGradShape = context->GetInputShape(0)->GetStorageShape();
        auto const xShapeVal = context->GetInputShape(1)->GetStorageShape();
        auto const yGradDtype = context->GetInputDesc(0)->GetDataType();

        dOut = yGradShape.GetDim(DIM0);
        hOut = yGradShape.GetDim(DIM1);
        wOut = yGradShape.GetDim(DIM2);
        ncNum = yGradShape.GetDim(DIM3);
        if (xShapeVal.GetDimNum() == X_DIMS_4){
            dIn=xShapeVal.GetDim(DIM1);
            hIn=xShapeVal.GetDim(DIM2);
            wIn=xShapeVal.GetDim(DIM3);
        } else if (xShapeVal.GetDimNum() == X_DIMS_5) {
            dIn=xShapeVal.GetDim(DIM2);
            hIn=xShapeVal.GetDim(DIM3);
            wIn=xShapeVal.GetDim(DIM4);
        }

        ncAlign = (ncNum + dataAlign - 1) / dataAlign * dataAlign;
        CalTilingKey(ubSize);
        return ge::GRAPH_SUCCESS;
    }

    void AdaptiveAvgPool3dGradTiling::CalTilingKey(uint32_t ubSize) {
        int64_t ALIGN_NUM = ALIGN_SIZE / yGradDSize;
        if (dIn % dOut == 0 && hIn % hOut == 0 && wIn % wOut == 0) {
            isAtomicAdd = 0;
        }
        if (isAtomicAdd == 1 && yGradDSize == BF_FP16_DSIZE) {
            useWorkspaceSize = ncNum * dIn * hIn * wIn * FP32_DSIZE;
        }
        if(yGradDSize == BF_FP16_DSIZE) {
            perCalcSize = ncAlign * yGradDSize + ncAlign * FP32_DSIZE  +  ncAlign * yGradDSize + ncAlign * FP32_DSIZE;        
        } else {
            perCalcSize = ncAlign * yGradDSize +  ncAlign * yGradDSize;
        }

        if (perCalcSize< ubSize){
            taskNum = dOut * hOut * wOut;
            yNumPerCalc= ubSize / perCalcSize;
            tilingKey = tilingKey * DTYPE_KEY_WEIGHT + NC_SMALL_KEY;
            ncAlignSliceLength = ncAlign;
            ncAlignSliceTail = ncAlign;
            ncSliceNum = 1;
        }else {
            tilingKey = tilingKey * DTYPE_KEY_WEIGHT + NC_LARGE_KEY;
            ncSliceNum = (perCalcSize - 1 + ubSize) / ubSize;
            ncAlignSliceLength= ncAlign / ncSliceNum / ALIGN_NUM * ALIGN_NUM;
            ncSliceNum = (ncNum - 1 + ncAlignSliceLength) / ncAlignSliceLength ;
            ncAlignSliceTail = ncNum - ncAlignSliceLength * (ncSliceNum - 1);
            taskNum = dOut * hOut * wOut * ncSliceNum; 
            yNumPerCalc= 1;
        }

        taskCoreUsed = taskNum > coreNum ? coreNum : taskNum;
        taskNumPerCore = (taskNum - 1 + taskCoreUsed)/taskCoreUsed;
        taskCoreUsed = (taskNum -1 + taskNumPerCore) / taskNumPerCore;
        taskNumLastCore = taskNum - taskNumPerCore * (taskCoreUsed-1);
        yNumPerCalc= yNumPerCalc > taskNumPerCore ? taskNumPerCore: yNumPerCalc;
        context->SetTilingKey(tilingKey);
    }
    
    ge::graphStatus AdaptiveAvgPool3dGradTiling::RunKernelTiling() {
        context->SetBlockDim(coreNum);
        TilingData.set_ncNum(ncNum);
        TilingData.set_dIn(dIn);
        TilingData.set_hIn(hIn);
        TilingData.set_wIn(wIn);
        TilingData.set_dOut(dOut);
        TilingData.set_hOut(hOut);
        TilingData.set_wOut(wOut);
        TilingData.set_taskCoreUsed(taskCoreUsed);
        TilingData.set_taskNumPerCore(taskNumPerCore);
        TilingData.set_taskNumLastCore(taskNumLastCore);
        TilingData.set_yNumPerCalc(yNumPerCalc);
        TilingData.set_ncSliceNum(ncSliceNum);  
        TilingData.set_ncAlignSliceLength(ncAlignSliceLength);
        TilingData.set_ncAlignSliceTail(ncAlignSliceTail);
        TilingData.set_isAtomicAdd(isAtomicAdd);
        TilingData.set_deterministicFlag(deterministicFlag);
        size_t* workspaces = context->GetWorkspaceSizes(1);
        workspaces[0] = useWorkspaceSize + sysWorkspaceSize + sysWorkspaceSize;
        TilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(TilingData.GetDataSize());
        return ge::GRAPH_SUCCESS;
    }
    
    bool AdaptiveAvgPool3dGradTiling::GetDataTypeKey(ge::DataType dataType) {
        switch (dataType) {
            case ge::DT_FLOAT16:
                yGradDSize = BF_FP16_DSIZE;
                tilingKey = FP16_DTYPE_KEY;
                dataAlign = blockBytes / yGradDSize;
                break;
            case ge::DT_BF16:
                yGradDSize = BF_FP16_DSIZE;
                tilingKey = BF16_DTYPE_KEY;
                dataAlign = blockBytes / yGradDSize;
                break;
            case ge::DT_FLOAT:
                yGradDSize = FP32_DSIZE;
                tilingKey = FP32_DTYPE_KEY;
                dataAlign = blockBytes / yGradDSize;
                break;
            default:
                return false;
        }
        return true;
    }

    static ge::graphStatus TilingFunc4AdaptiveAvgPool3dGrad(gert::TilingContext* context) {
        AdaptiveAvgPool3dGradTiling tilingObject(context);
        if (tilingObject.Init()!=ge::GRAPH_SUCCESS){
            return ge::GRAPH_FAILED;
        }
        return tilingObject.RunKernelTiling();
    }
    
    IMPL_OP_OPTILING(AdaptiveAvgPool3dGrad)
        .Tiling(TilingFunc4AdaptiveAvgPool3dGrad);
} // namespace optiling
