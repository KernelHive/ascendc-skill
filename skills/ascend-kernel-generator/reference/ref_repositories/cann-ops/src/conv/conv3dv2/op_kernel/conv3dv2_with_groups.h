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
 * \file conv3dv2_with_groups.h
 * \brief
 */

#ifndef CONV3DV2_WITH_GROUPS_H
#define CONV3DV2_WITH_GROUPS_H

#include "conv3dv2.h"

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class CONV_CFG>
class KernelConv3DV2WithGroups : public KernelConv3DV2<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CONV_CFG> {
public:
    __aicore__ inline KernelConv3DV2WithGroups() {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR scale, GM_ADDR offset,
        GM_ADDR y, GM_ADDR workspace, const Conv3DTilingData *allTilingData)
    {
        this->InitTilingData(allTilingData);
        this->InitC1N1();
        bool isRealDim = InitSingleCoreData();
        if (!isRealDim) [[unlikely]] {
            this->normalInit = false;
        }
        InitBuffer(x, filter, bias, y);
    }

    __aicore__ inline void Process()
    {
        if (!this->normalInit || this->blockIdx > this->blockDim) [[unlikely]] {
            return;
        }
        Conv3DV2KernelImpl();
    }

protected:
    __aicore__ inline bool InitSingleCoreDataWithBatch()
    {
        const uint64_t dataPerBatchDim = this->conv3dRunInfo->groupDim * this->conv3dRunInfo->mDim *
                                         this->conv3dRunInfo->nDim * this->conv3dRunInfo->doDim;
        DataToFill batchStruct(this->singleCoreBatch, this->batchIdxStart, this->isBatchDimTail);
        bool isRealDim = this->CountIdxTail(dataPerBatchDim,
            this->conv3dRunInfo->batchDim,
            this->conv3dRunInfo->batch,
            this->conv3dRunInfo->batch,
            batchStruct);
        if (!isRealDim) [[unlikely]] {
            return false;
        }

        return true;
    }

    __aicore__ inline bool InitSingleCoreDataWithGroupOpt()
    {
        const uint64_t dataPerGroupOptDim =
            this->conv3dRunInfo->doDim * this->conv3dRunInfo->nDim * this->conv3dRunInfo->mDim;
        DataToFill groupOptStruct(singleCoreGroupOpt, groupOptIdxStart, isGroupOptDimTail);
        bool isRealDim = this->CountIdxTail(dataPerGroupOptDim,
            this->conv3dRunInfo->groupDim,
            this->conv3dApiTiling->groupOpt,
            this->conv3dApiTiling->groupOpt,
            groupOptStruct);
        if (!isRealDim) [[unlikely]] {
            return false;
        }

        return true;
    }

    __aicore__ inline bool InitSingleCoreDataWithDout()
    {
        const uint64_t dataPerDoDim = this->conv3dRunInfo->nDim * this->conv3dRunInfo->mDim;
        DataToFill doStruct(this->singleCoreDout, this->doIdxStart, this->isDoDimTail);
        bool isRealDim = this->CountIdxTail(
            dataPerDoDim, this->conv3dRunInfo->doDim, this->conv3dRunInfo->dout, this->conv3dRunInfo->dout, doStruct);
        if (!isRealDim) [[unlikely]] {
            return false;
        }

        return true;
    }

    __aicore__ inline bool InitSingleCoreDataWithCout()
    {
        DataToFill nStruct(this->singleCoreN, this->nIdxStart, this->isNDimTail);
        bool isRealDim = this->CountIdxTail(this->conv3dRunInfo->mDim,
            this->conv3dRunInfo->nDim,
            AlignB(this->conv3dApiTiling->coutOpt, this->n0),
            this->conv3dApiTiling->coutOpt,
            nStruct);
        if (!isRealDim) [[unlikely]] {
            return false;
        }

        return true;
    }

    __aicore__ inline bool InitSingleCoreDataWithM()
    {
        DataToFill mStruct(this->singleCoreM, this->mIdxStart, this->isMDimTail);
        const uint64_t totalM = this->conv3dRunInfo->wout * this->conv3dRunInfo->hout;
        bool isRealDim = this->CountIdxTail(1, this->conv3dRunInfo->mDim, totalM, totalM, mStruct);  // dataPerMDim = 1
        if (!isRealDim) [[unlikely]] {
            return false;
        }

        return true;
    }

    __aicore__ inline bool InitSingleCoreData()
    {
        if (!InitSingleCoreDataWithBatch() || !InitSingleCoreDataWithGroupOpt() || !InitSingleCoreDataWithDout()) {
            return false;
        }

        if (!InitSingleCoreDataWithCout() || !InitSingleCoreDataWithM()) {
            return false;
        }

        uint64_t preCoreSumCout = groupOptIdxStart * this->conv3dApiTiling->coutOpt + this->nIdxStart;
        uint64_t curCoreSumCout = this->conv3dApiTiling->singleCoreGroupOpt * this->conv3dApiTiling->singleCoreCo;
        if (preCoreSumCout >= this->conv3dRunInfo->cout) {
            ASC_OP_LOGD("[Conv3dv2WithGroups] No real data exists in groupopt. Skip the processing. groupOptIdxStart "
                        "%d, nIdxStart %d.",
                groupOptIdxStart,
                this->nIdxStart);
            return false;
        }
        uint64_t groupDimTailCout = AlignB(this->conv3dRunInfo->cout % this->conv3dApiTiling->coutOpt, this->c0Out);
        if (isGroupOptDimTail && groupDimTailCout != 0 && this->nIdxStart >= groupDimTailCout) {
            singleCoreGroupOpt -= 1;
            ASC_OP_LOGD(
                "[IterateAllWithGroups] No real data exists in groupTail. Skip the groupTail processing. nIdxStart "
                "%d. \n",
                this->nIdxStart);
            return true;
        }
        if (isGroupOptDimTail) {
            singleCoreCinTail = this->conv3dRunInfo->cin % this->conv3dApiTiling->cinOpt;
            if (this->isNDimTail || preCoreSumCout + curCoreSumCout >= this->conv3dRunInfo->cout) {
                singleCoreCoutTail =
                    (this->conv3dRunInfo->cout % this->conv3dApiTiling->coutOpt) % this->conv3dApiTiling->singleCoreCo;
            }
            return true;
        }

        return true;
    }

    __aicore__ inline void InitBuffer(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y)
    {
        this->fmapOneBatchSize =
            this->conv3dRunInfo->din * this->c1In * this->conv3dRunInfo->hin * this->conv3dRunInfo->win * this->c0In;
        fmapOneGroupOptSize = this->conv3dApiTiling->cinOpt * this->conv3dRunInfo->hin * this->conv3dRunInfo->win;
        weightOneGroupOptSize = this->conv3dApiTiling->cinOpt * this->conv3dApiTiling->kernelD *
                                this->conv3dApiTiling->kernelH * this->conv3dApiTiling->kernelW *
                                this->conv3dApiTiling->coutOpt;
        this->outputOneBatchSize =
            this->conv3dRunInfo->dout * this->c1Out * this->conv3dRunInfo->hout * this->conv3dRunInfo->wout * this->c0Out;
        outputOneGroupOptSize = this->conv3dApiTiling->coutOpt * this->conv3dRunInfo->hout * this->conv3dRunInfo->wout;

        int64_t diIdxStart = this->doIdxStart * this->conv3dRunInfo->strideD - this->conv3dRunInfo->padHead;
        int64_t hiIdxStart =
            (this->mIdxStart / this->conv3dRunInfo->wout) * this->conv3dRunInfo->strideH - this->conv3dRunInfo->padUp;

        uint64_t fmStartAddr =
            this->batchIdxStart * this->fmapOneBatchSize +
            this->Max(diIdxStart, 0) * this->c1In * this->conv3dRunInfo->hin * this->conv3dRunInfo->win * this->c0In +
            groupOptIdxStart * fmapOneGroupOptSize + this->Max(hiIdxStart, 0) * this->conv3dRunInfo->win * this->c0In;
        uint64_t weightStartAddr = groupOptIdxStart * weightOneGroupOptSize + this->nIdxStart * this->c0K;
        uint64_t outputStartAddr =
            this->batchIdxStart * this->outputOneBatchSize +
            this->doIdxStart * this->c1Out * this->conv3dRunInfo->hout * this->conv3dRunInfo->wout * this->c0Out +
            groupOptIdxStart * outputOneGroupOptSize +
            this->nIdxStart * this->conv3dRunInfo->hout * this->conv3dRunInfo->wout + this->mIdxStart * this->c0Out;
        ASC_OP_LOGD("[Conv3dv2WithGroups] fmStartAddr %d weightStartAddr %d outputStartAddr %d.\n",
            fmStartAddr,
            weightStartAddr,
            outputStartAddr);

        this->fmapGm.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(x + fmStartAddr * sizeof(A_T)));
        this->filterGm.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(filter + weightStartAddr * sizeof(B_T)));
        this->outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(y + outputStartAddr * sizeof(C_T)));
        if (this->conv3dRunInfo->hasBias) {
            uint64_t biasStartAddr = groupOptIdxStart * this->conv3dApiTiling->coutOpt + this->nIdxStart;
            ASC_OP_LOGD("[Conv3dv2WithGroups] biasStartAddr %d.\n", biasStartAddr);
            this->biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ BIAS_T *>(bias + biasStartAddr * sizeof(BIAS_T)));
        }
    }

    __aicore__ inline void Conv3DV2KernelImpl()
    {
        this->conv.Init(this->conv3dApiTiling);
        if (this->isDoDimTail || this->isNDimTail || this->isMDimTail || isGroupOptDimTail) [[unlikely]] {
            this->conv.SetSingleOutputShape(1,
                this->singleCoreN,
                this->singleCoreDout,
                this->singleCoreM,
                singleCoreGroupOpt);
        }

        int64_t diIdxStart = this->doIdxStart * this->conv3dRunInfo->strideD;
        int64_t hiIdxStart = (this->mIdxStart / this->conv3dRunInfo->wout) * this->conv3dRunInfo->strideH;
        ASC_OP_LOGD("[Conv3dv2WithGroups] doIdxStart %d mIdxStart %d diIdxStart %d hiIdxStart %d.\n",
            this->doIdxStart,
            this->mIdxStart,
            diIdxStart,
            hiIdxStart);

        this->conv.SetFmapStartPosition(this->Max(diIdxStart, 0), this->mIdxStart, 0);
        for (uint64_t batchIter = 0; batchIter < this->singleCoreBatch; ++batchIter) {
            this->conv.SetGroupOptInfo(singleCoreCinTail, singleCoreCoutTail, isGroupOptDimTail);
            this->conv.SetFmap(this->fmapGm[batchIter * this->fmapOneBatchSize]);
            this->conv.SetWeight(this->filterGm);
            if (this->conv3dRunInfo->hasBias) {
                this->conv.SetBias(this->biasGm);
            }
            this->conv.IterateAll(this->outputGm[batchIter * this->outputOneBatchSize]);
            this->conv.End();
        }
    }

protected:
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BIAS_T = typename BIAS_TYPE::T;

    uint64_t singleCoreGroupOpt = 0;
    uint64_t groupOptIdxStart = 0;
    bool isGroupOptDimTail = false;
    uint64_t fmapOneGroupOptSize = 0;
    uint64_t weightOneGroupOptSize = 0;
    uint64_t outputOneGroupOptSize = 0;
    uint64_t singleCoreCinTail = 0;
    uint64_t singleCoreCoutTail = 0;
};

#endif
