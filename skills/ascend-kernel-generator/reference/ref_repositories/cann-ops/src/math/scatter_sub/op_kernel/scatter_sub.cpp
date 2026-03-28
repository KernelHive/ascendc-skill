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
 * @file add_custom.cpp
 */
#include "kernel_operator.h"
constexpr int32_t BUFFER_NUM = 1;                                     // tensor num for each queue

template<typename TYPE_VAR, typename TYPE_INDICES, typename TYPE_UPDATES> class KernelScatterSubAligned {
    using T = TYPE_VAR;
public:
    __aicore__ inline KernelScatterSubAligned() {}
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, uint32_t alignNum,
        uint32_t lastDim, uint32_t indicesLength, uint32_t var1stDim, uint32_t firstTiling)
    {        
        this->tileNum = indicesLength;
        this->indicesLength = indicesLength;
        this->lastDim = lastDim;
        this->var1stDim = var1stDim;
        this->firstTiling = firstTiling;

        varGm.SetGlobalBuffer((__gm__ TYPE_VAR*)var, var1stDim * lastDim);
        indicesGm.SetGlobalBuffer((__gm__ TYPE_INDICES*)indices, indicesLength);
        updatesGm.SetGlobalBuffer((__gm__ TYPE_UPDATES*)updates , indicesLength * lastDim);

        pipe.InitBuffer(inQueueVar, BUFFER_NUM, firstTiling * sizeof(TYPE_VAR));

        pipe.InitBuffer(inQueueUpdates, BUFFER_NUM,  firstTiling * sizeof(TYPE_UPDATES));
        pipe.InitBuffer(outQueueVar, BUFFER_NUM, firstTiling * sizeof(TYPE_VAR));

        pipe.InitBuffer(varBuf, firstTiling * sizeof(half));
        pipe.InitBuffer(updatesBuf, firstTiling * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount; i++) {
            uint32_t total = lastDim;
            uint32_t count = lastDim / firstTiling;
            uint32_t left = lastDim % firstTiling;
            for (int ii = 0; ii < count ;ii++) {
                CopyIn(i, ii, firstTiling);
                Compute(i, ii, firstTiling);
                CopyOut(i, ii, firstTiling);
            }
            if (left !=0) {
                CopyIn(i, count, left);
                Compute(i, count, left);
                CopyOut(i, count, left);
            }
        }
    }
private:
    __aicore__ inline void CopyIn(int32_t progress, int32_t lastDimProgress, uint32_t tilingNum)
    {
        AscendC::LocalTensor<TYPE_UPDATES> updatesLocal = inQueueUpdates.AllocTensor<TYPE_UPDATES>();
        AscendC::LocalTensor<TYPE_VAR> varLocal = inQueueVar.AllocTensor<TYPE_VAR>();

        AscendC::DataCopy(updatesLocal, updatesGm[progress * lastDim + lastDimProgress * firstTiling], tilingNum);
        TYPE_INDICES indice = indicesGm.GetValue(progress);
        AscendC::DataCopy(varLocal, varGm[indice * lastDim + lastDimProgress * firstTiling], tilingNum);

        inQueueUpdates.EnQue(updatesLocal);
        inQueueVar.EnQue(varLocal);
    }
    __aicore__ inline void Compute(int32_t progress, int32_t lastDimProgress, uint32_t tilingNum)
    {
        AscendC::LocalTensor<TYPE_UPDATES> updatesLocal = inQueueUpdates.DeQue<TYPE_UPDATES>();
        AscendC::LocalTensor<TYPE_VAR> varLocal = inQueueVar.DeQue<TYPE_VAR>();
        AscendC::LocalTensor<TYPE_VAR> outVarLocal = outQueueVar.AllocTensor<TYPE_VAR>();
        
        if constexpr (std::is_same_v<T, int8_t>) {
            AscendC::LocalTensor<half> varBufLocal = varBuf.Get<half>();
            AscendC::LocalTensor<half> updatesBufLocal = updatesBuf.Get<half>();
        
            AscendC::Cast(varBufLocal, varLocal, AscendC::RoundMode::CAST_NONE, tilingNum);
            AscendC::Cast(updatesBufLocal, updatesLocal, AscendC::RoundMode::CAST_NONE, tilingNum);

            AscendC::Sub(varBufLocal, varBufLocal, updatesBufLocal, tilingNum);
            AscendC::Cast(outVarLocal, varBufLocal, AscendC::RoundMode::CAST_ROUND, tilingNum);
        }else {
            AscendC::Sub(outVarLocal, varLocal, updatesLocal, tilingNum);
        }

        inQueueUpdates.FreeTensor(updatesLocal);
        inQueueVar.FreeTensor(varLocal);
        outQueueVar.EnQue(outVarLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress, int32_t lastDimProgress, uint32_t tilingNum)
    {
        AscendC::LocalTensor<TYPE_VAR> outVarLocal = outQueueVar.DeQue<TYPE_VAR>();
        TYPE_INDICES indice = indicesGm.GetValue(progress);
        AscendC::DataCopy(varGm[indice * lastDim + lastDimProgress * firstTiling], outVarLocal, tilingNum);
        outQueueVar.FreeTensor(outVarLocal);
    }
private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueUpdates;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueVar;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueVar;

    AscendC::TBuf<AscendC::TPosition::VECCALC> varBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> updatesBuf;

    AscendC::GlobalTensor<TYPE_VAR> varGm;
    AscendC::GlobalTensor<TYPE_UPDATES> updatesGm;
    AscendC::GlobalTensor<TYPE_INDICES> indicesGm;

    uint32_t tileNum;

    uint32_t indicesLength;
    uint32_t lastDim;
    uint32_t var1stDim;
    uint32_t firstTiling;
};

template<typename TYPE_VAR, typename TYPE_INDICES, typename TYPE_UPDATES> class KernelScatterSubUnAligned {
using T = TYPE_VAR;
public:
    __aicore__ inline KernelScatterSubUnAligned() {}
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, uint32_t alignNum,
        uint32_t lastDim, uint32_t indicesLength, uint32_t var1stDim, uint32_t firstTiling)
    {        
        this->tileNum = indicesLength;
        this->indicesLength = indicesLength;
        this->lastDim = lastDim;
        this->var1stDim = var1stDim;
        this->firstTiling = firstTiling;

        varGm.SetGlobalBuffer((__gm__ TYPE_VAR*)var, var1stDim * lastDim);
        indicesGm.SetGlobalBuffer((__gm__ TYPE_INDICES*)indices, indicesLength);
        updatesGm.SetGlobalBuffer((__gm__ TYPE_UPDATES*)updates , indicesLength * lastDim);        
    }
    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < indicesLength; i ++) {
            for (int32_t ii = 0; ii < lastDim; ii ++) {
                TYPE_INDICES indice = indicesGm.GetValue(i);
                TYPE_VAR var = varGm.GetValue(indice * lastDim + ii);
                TYPE_UPDATES update = updatesGm.GetValue(i * lastDim + ii);
                if constexpr (std::is_same_v<T, half>) {
                    // half转float
                    int16_t varInt16 = *(int16_t*)&var;
                    int32_t fltInt32 = ((varInt16 & 0x8000) << 16);
                    fltInt32 |= ((varInt16 & 0x7fff) << 13) + 0x38000000;
                    float varFp32 = *(float*)&fltInt32;

                    int16_t updateInt16 = *(int16_t*)&update;
                    fltInt32 = ((updateInt16 & 0x8000) << 16);
                    fltInt32 |= ((updateInt16 & 0x7fff) << 13) + 0x38000000;
                    float updateFp32 = *(float*)&fltInt32;
                    float res = varFp32 - updateFp32;

                    // float转half
                    int16_t fltInt16;
                    fltInt32 = *(int32_t*)&res;
                    fltInt16 = ((fltInt32 & 0x7fffffff) >> 13) - (0x38000000 >> 13);
                    fltInt16 |= ((fltInt32 & 0x80000000) >> 16);
                    half resHalf = *(half*)&fltInt16;

                    varGm.SetValue(indice * lastDim + ii, resHalf);
                }else {
                    varGm.SetValue(indice * lastDim + ii, var - update);
                }  
            }
        }
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueUpdates;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueVar;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueVar;

    AscendC::GlobalTensor<TYPE_VAR> varGm;
    AscendC::GlobalTensor<TYPE_UPDATES> updatesGm;
    AscendC::GlobalTensor<TYPE_INDICES> indicesGm;

    uint32_t tileNum;
    uint32_t indicesLength;
    uint32_t lastDim;
    uint32_t var1stDim;
    uint32_t firstTiling;
};
extern "C" __global__ __aicore__ void scatter_sub(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    if (tilingData.firstTiling == 0) {
        KernelScatterSubUnAligned<DTYPE_VAR, DTYPE_INDICES, DTYPE_UPDATES> op;
        op.Init(var, indices, updates, tilingData.alignNum, tilingData.lastDim, tilingData.indicesLength, tilingData.var1stDim, tilingData.firstTiling);
        op.Process();
    }else {
        KernelScatterSubAligned<DTYPE_VAR, DTYPE_INDICES, DTYPE_UPDATES> op;
        op.Init(var, indices, updates, tilingData.alignNum, tilingData.lastDim, tilingData.indicesLength, tilingData.var1stDim, tilingData.firstTiling);
        op.Process();
    }
}