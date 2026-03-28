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
 * \file conv3dv2.h
 * \brief
 */

#ifndef CONV3DV2_H
#define CONV3DV2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "conv3d_api.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace conv3d;

constexpr uint8_t SINGLE_BLOCK_SIZE = 32;

template <ConvL0PingPong pingPong_T, ConvBL1ByPass bl1BypassFlag_T, GroupConvType groupConvType_T,
          OutputOrder outputOrder_T, QuantType quantType_T, LoadChannelType loadChannelType_T>
struct Conv3DV2Param : public ConvParam {
    __aicore__ inline Conv3DV2Param() {};
    constexpr static int8_t outputOrder = static_cast<int8_t>(outputOrder_T);
    constexpr static int8_t l0pingpong = static_cast<int8_t>(pingPong_T);
    constexpr static int8_t bl1bypass = static_cast<int8_t>(bl1BypassFlag_T);
    constexpr static int8_t groupConvType = static_cast<int8_t>(groupConvType_T);
    constexpr static int8_t quantType = static_cast<int8_t>(quantType_T);
    constexpr static int8_t loadChannelType = static_cast<int8_t>(loadChannelType_T);
};

struct DataToFill {
    __aicore__ inline DataToFill(uint64_t &singleCoreDim_, uint64_t &dimIdxStart_, bool &isDimTail_)
        : singleCoreDim(singleCoreDim_), dimIdxStart(dimIdxStart_), isDimTail(isDimTail_) {}

    uint64_t &singleCoreDim;
    uint64_t &dimIdxStart;
    bool &isDimTail;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class CONV_CFG>
class KernelConv3DV2 {
public:
    __aicore__ inline KernelConv3DV2() {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR scale, GM_ADDR offset,
        GM_ADDR y, GM_ADDR workspace, const Conv3DTilingData *allTilingData)
    {
        if ASCEND_IS_AIV {  
            blockIdx = blockIdx / subblocknum;
            subblockIdx = GetSubBlockIdx();
        }
        
        InitTilingData(allTilingData);
        InitC1N1();
        bool isRealDim = InitSingleCoreData();
        if (!isRealDim) [[unlikely]] {
            normalInit = false;
        }
        InitBuffer(x, filter, bias, y, scale, workspace);
    }

    __aicore__ inline void Process()
    {
        if (!normalInit || blockIdx >= blockDim) [[unlikely]] {
            return;
        }
        Conv3DV2KernelImpl();
    }

protected:
    __aicore__ inline void InitTilingData(const Conv3DTilingData *allTilingData)
    {
        this->allTilingData = allTilingData;
        this->conv3dRunInfo = &(allTilingData->conv3dRunInfo);
        this->conv3dApiTiling = &(allTilingData->conv3dApiTiling);

        blockDim = this->conv3dRunInfo->mDim * this->conv3dRunInfo->nDim * this->conv3dRunInfo->groupDim * this->conv3dRunInfo->doDim *
                   this->conv3dRunInfo->batchDim;
    }

    __aicore__ inline uint64_t CeilDiv(const uint64_t &num, const uint64_t &by)
    {
        if (by == 0UL) {
            return num;
        }
        return (num + by - 1) / by;
    }

    __aicore__ inline void InitC1N1()
    {
        c1In = CeilDiv(conv3dRunInfo->cin, c0In);
        c1Out = CeilDiv(conv3dRunInfo->cout, c0Out);
        n1 = CeilDiv(conv3dRunInfo->cout, n0);
    }

    __aicore__ inline bool CountIdxTail(const uint64_t &dataPerDim, const uint64_t &dim, const uint64_t &wholeDim,
        const uint64_t &realWholeDim, DataToFill &curStruct)
    {
        if (dataPerDim == 0 || dim == 0) {
            return false;
        }
        const uint64_t dimIdx = (blockIdx / dataPerDim) % dim;
        const uint64_t maxDimPerCore = CeilDiv(wholeDim, dim);
        const uint64_t realDim = CeilDiv(wholeDim, maxDimPerCore);
        if (dimIdx >= realDim) [[unlikely]] {
            return false;
        }

        curStruct.isDimTail = (dimIdx == (realDim - 1));
        curStruct.singleCoreDim = !curStruct.isDimTail ? maxDimPerCore : realWholeDim - (realDim - 1) * maxDimPerCore;
        curStruct.dimIdxStart = dimIdx * maxDimPerCore;
        return true;
    }

    __aicore__ inline bool InitSingleCoreDataWithBatch()
    {
        const uint64_t dataPerBatchDim = conv3dRunInfo->mDim * conv3dRunInfo->nDim * conv3dRunInfo->doDim;
        DataToFill batchStruct(singleCoreBatch, batchIdxStart, isBatchDimTail);
        bool isRealDim = CountIdxTail(
            dataPerBatchDim, conv3dRunInfo->batchDim, conv3dRunInfo->batch, conv3dRunInfo->batch, batchStruct);
        if (!isRealDim) [[unlikely]] {
            return false;
        }

        return true;
    }

    __aicore__ inline bool InitSingleCoreDataWithDout()
    {
        const uint64_t dataPerDoDim = conv3dRunInfo->nDim * conv3dRunInfo->mDim;
        DataToFill doStruct(singleCoreDout, doIdxStart, isDoDimTail);
        bool isRealDim =
            CountIdxTail(dataPerDoDim, conv3dRunInfo->doDim, conv3dRunInfo->dout, conv3dRunInfo->dout, doStruct);
        if (!isRealDim) [[unlikely]] {
            return false;
        }

        return true;
    }

    __aicore__ inline bool InitSingleCoreDataWithCout()
    {
        DataToFill nStruct(singleCoreN, nIdxStart, isNDimTail);
        bool isRealDim = CountIdxTail(conv3dRunInfo->mDim, conv3dRunInfo->nDim, n1 * n0, conv3dRunInfo->cout, nStruct);
        if (!isRealDim) [[unlikely]] {
            return false;
        }

        return true;
    }

    __aicore__ inline bool InitSingleCoreDataWithM()
    {
        DataToFill mStruct(singleCoreM, mIdxStart, isMDimTail);
        const uint64_t totalM = conv3dRunInfo->wout * conv3dRunInfo->hout;
        bool isRealDim = CountIdxTail(1, conv3dRunInfo->mDim, totalM, totalM, mStruct);  // dataPerMDim = 1
        if (!isRealDim) [[unlikely]] {
            return false;
        }

        return true;
    }

    __aicore__ inline bool InitSingleCoreData()
    {
        if (!InitSingleCoreDataWithBatch() || !InitSingleCoreDataWithDout() ||
            !InitSingleCoreDataWithCout() || !InitSingleCoreDataWithM()) {
            return false;
        }

        return true;
    }

    __aicore__ inline int64_t Max(const int64_t &left, const int64_t &right)
    {
        return left > right ? left : right;
    }

    __aicore__ inline void InitBuffer(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y, GM_ADDR scale, GM_ADDR workspace)
    {
        fmapOneBatchSize = conv3dRunInfo->din * c1In * conv3dRunInfo->hin * conv3dRunInfo->win * c0In;
        outputOneBatchSize = conv3dRunInfo->dout * c1Out * conv3dRunInfo->hout * conv3dRunInfo->wout * c0Out;

        int64_t diIdxStart = doIdxStart * conv3dRunInfo->strideD - conv3dRunInfo->padHead;
        int64_t hiIdxStart = (mIdxStart / conv3dRunInfo->wout) * conv3dRunInfo->strideH - conv3dRunInfo->padUp;

        uint64_t fmStartAddr = batchIdxStart * fmapOneBatchSize +
                               Max(diIdxStart, 0) * c1In * conv3dRunInfo->hin * conv3dRunInfo->win * c0In +
                               Max(hiIdxStart, 0) * conv3dRunInfo->win * c0In;
        uint64_t weightStartAddr = nIdxStart * c0K;
        uint64_t outputStartAddr = batchIdxStart * outputOneBatchSize +
                                   doIdxStart * c1Out * conv3dRunInfo->hout * conv3dRunInfo->wout * c0Out +
                                   nIdxStart * conv3dRunInfo->hout * conv3dRunInfo->wout + mIdxStart * c0Out;
        ASC_OP_LOGD("[InitBuffer] fmStartAddr %d weightStartAddr %d outputStartAddr %d.\n",
            fmStartAddr,
            weightStartAddr,
            outputStartAddr);

        fmapGm.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(x + fmStartAddr * sizeof(A_T)));
        filterGm.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(filter + weightStartAddr * sizeof(B_T)));
        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(y + outputStartAddr * sizeof(C_T)));
        if (conv3dRunInfo->hasBias) {
            uint64_t biasStartAddr = nIdxStart;
            ASC_OP_LOGD("[InitBuffer] biasStartAddr %d.\n", biasStartAddr);
            biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ BIAS_T *>(bias + biasStartAddr * sizeof(BIAS_T)));
        }
      
        if constexpr (CONV_CFG::quantType == static_cast<int8_t>(QuantType::PER_CHANNEL_NO_OFFSET)) {
            uint64_t wsStartAddr = (blockIdx * 4) * conv3dApiTiling->mL0 * conv3dApiTiling->nL0;
            workspaceGm.SetGlobalBuffer(reinterpret_cast<__gm__ typename conv::GetDstType<A_T>::Type *>(workspace + wsStartAddr * sizeof(typename conv::GetDstType<A_T>::Type))); 
            scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scale + nIdxStart * sizeof(float)));
        }
    }

    __aicore__ inline void Conv3DV2KernelImpl()
    {
        conv.Init(conv3dApiTiling);
        if (isDoDimTail || isNDimTail || isMDimTail) [[unlikely]] {
            conv.SetSingleOutputShape(1, singleCoreN, singleCoreDout, singleCoreM, 0);
        }

        int64_t diIdxStart = doIdxStart * conv3dRunInfo->strideD;
        int64_t hiIdxStart = (mIdxStart / conv3dRunInfo->wout) * conv3dRunInfo->strideH;
        ASC_OP_LOGD("[Conv3DV2KernelImpl] doIdxStart %d mIdxStart %d diIdxStart %d hiIdxStart %d.\n",
            doIdxStart,
            mIdxStart,
            diIdxStart,
            hiIdxStart);

        conv.SetFmapStartPosition(Max(diIdxStart, 0), mIdxStart, 0);
        conv.SetWeight(filterGm);
        
        if (conv3dRunInfo->hasBias) {
            conv.SetBias(biasGm);
        }

        conv.SetWorkspace(workspaceGm);
        conv.SetVecScale(scaleGm);
        conv.SetSubBlockIdx(subblockIdx);

        conv.LoadTotalCoreChannel();

        conv.BeginSetCrossFlag();

        for (uint64_t batchIter = 0; batchIter < singleCoreBatch; ++batchIter) {
            conv.SetFmap(fmapGm[batchIter * fmapOneBatchSize]);
            conv.IterateAll(outputGm[batchIter * outputOneBatchSize]);
            conv.End();
        }

        conv.EndWaitCrossFlag();
        conv.FreeTotalCoreChannel();
    }

protected:
    // Conv3D API
    Conv3d<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CONV_CFG> conv;
    // Get dtype
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BIAS_T = typename BIAS_TYPE::T;

    // Input and output tensor declare
    GlobalTensor<A_T> fmapGm;
    GlobalTensor<B_T> filterGm;
    GlobalTensor<C_T> outputGm;
    GlobalTensor<BIAS_T> biasGm;
    GlobalTensor<typename conv::GetDstType<A_T>::Type> workspaceGm; 
    GlobalTensor<float> scaleGm;

    // Tiling data
    const TConv3DTiling *conv3dApiTiling;
    const Conv3DRunInfo *conv3dRunInfo;
    const Conv3DTilingData *allTilingData;

    uint32_t blockIdx = GetBlockIdx();
    uint32_t subblocknum = GetSubBlockNum();
    uint32_t subblockIdx;

    uint32_t blockDim;

    // Single core data
    uint64_t singleCoreBatch;
    uint64_t singleCoreDout;
    uint64_t singleCoreN;
    uint64_t singleCoreM;

    bool isBatchDimTail;
    bool isDoDimTail;
    bool isNDimTail;
    bool isMDimTail;

    bool normalInit = true;

    uint64_t fmapOneBatchSize;
    uint64_t outputOneBatchSize;

    uint64_t batchIdxStart;
    uint64_t doIdxStart;
    uint64_t nIdxStart;
    uint64_t mIdxStart;

    uint64_t c1In;
    uint64_t c1Out;
    uint64_t n1;

    static constexpr uint8_t n0 = 16;
    static constexpr uint8_t c0In = SINGLE_BLOCK_SIZE / sizeof(A_T);
    static constexpr uint8_t c0K = SINGLE_BLOCK_SIZE / sizeof(B_T);
    static constexpr uint8_t c0Out = SINGLE_BLOCK_SIZE / sizeof(C_T);
};

#endif
