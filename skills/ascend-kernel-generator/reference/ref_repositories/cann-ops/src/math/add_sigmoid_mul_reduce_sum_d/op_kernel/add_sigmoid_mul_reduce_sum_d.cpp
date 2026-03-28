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
 * @file add_sigmoid_mul_reduce_sum_d.cpp
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

namespace AscendC
{
    template <typename T>
    class KernelAddSigmoidMulReduceSumD
    {
    public:
        __aicore__ inline KernelAddSigmoidMulReduceSumD() {}
        __aicore__ inline void InitData(uint32_t formerCoreNum,
            uint32_t formerCoreLength,
            uint32_t formerTileNum,
            uint32_t formerTileLength,
            uint32_t formerLastTileLength,
            uint32_t tailCoreNum,
            uint32_t tailCoreLength,
            uint32_t tailTileNum,
            uint32_t tailTileLength,
            uint32_t tailLastTileLength,
            int32_t addInput0Dim1234Length,
            int32_t addInput0Dim14Length,
            int32_t addInput0Dim23Length,
            int32_t addInput0Dim1Length,
            int32_t addInput0Dim234Length)
        {
            this->formerCoreNum = formerCoreNum;
            this->formerCoreLength = formerCoreLength;
            this->formerTileNum = formerTileNum;
            this->formerTileLength = formerTileLength;
            this->formerLastTileLength = formerLastTileLength;

            this->tailCoreNum = tailCoreNum;
            this->tailCoreLength = tailCoreLength;
            this->tailTileNum = tailTileNum;
            this->tailTileLength = tailTileLength;
            this->tailLastTileLength = tailLastTileLength;

            this->addInput0Dim1234Length = addInput0Dim1234Length;
            this->addInput0Dim14Length = addInput0Dim14Length;
            this->addInput0Dim23Length = addInput0Dim23Length;
            this->addInput0Dim1Length = addInput0Dim1Length;
            this->addInput0Dim234Length = addInput0Dim234Length;
            this->blockOffset = this->addInput0Dim14Length / (this->size_32 / sizeof(T));
        }

        __aicore__ inline void Init(GM_ADDR add_0_input0, GM_ADDR add_0_input1, GM_ADDR mul_0_input1, 
            GM_ADDR mult_1_input1, GM_ADDR mult_2_input1, GM_ADDR out, GM_ADDR workspace)
        {
            this->blockIdx = GetBlockIdx();

            mul0Input1Global.SetGlobalBuffer((__gm__ T *)mul_0_input1, 1 * sizeof(T));
            this->ignoreIndex = mul0Input1Global.GetValue(0);
            if (this->blockIdx < this->formerCoreNum)
            {
                this->tileLength = this->formerTileLength;
                this->lastTileLength = this->formerLastTileLength;
                this->tileNum = this->formerTileNum;
                this->outOffset = this->blockIdx * this->formerCoreLength * this->addInput0Dim14Length;
                add0Input0Global.SetGlobalBuffer((__gm__ T *)add_0_input0 + this->blockIdx * this->formerCoreLength * this->addInput0Dim1234Length, this->formerCoreLength * this->addInput0Dim1234Length);
                mult1Input1Global.SetGlobalBuffer((__gm__ T *)mult_1_input1 + this->blockIdx * this->formerCoreLength * this->addInput0Dim23Length, this->formerCoreLength * this->addInput0Dim23Length);
                mult2Input1Global.SetGlobalBuffer((__gm__ T *)mult_2_input1 + this->blockIdx * this->formerCoreLength * this->addInput0Dim1234Length, this->formerCoreLength * this->addInput0Dim1234Length);
            }
            else
            {
                this->tileLength = this->tailTileLength;
                this->lastTileLength = this->tailLastTileLength;
                this->tileNum = this->tailTileNum;
                this->outOffset = this->formerCoreNum * this->formerCoreLength * this->addInput0Dim14Length 
                        + (this->blockIdx - this->formerCoreNum) * this->tailCoreLength * this->addInput0Dim14Length;
                add0Input0Global.SetGlobalBuffer((__gm__ T *)add_0_input0 + this->formerCoreNum 
                        * this->formerCoreLength * this->addInput0Dim1234Length + (this->blockIdx - this->formerCoreNum) 
                        * this->tailCoreLength * this->addInput0Dim1234Length, this->tailCoreLength * this->addInput0Dim1234Length);
                mult1Input1Global.SetGlobalBuffer((__gm__ T *)mult_1_input1 + this->formerCoreNum * this->formerCoreLength 
                        * this->addInput0Dim23Length + (this->blockIdx - this->formerCoreNum) * this->tailCoreLength 
                        * this->addInput0Dim23Length, this->tailCoreLength * this->addInput0Dim23Length);
                mult2Input1Global.SetGlobalBuffer((__gm__ T *)mult_2_input1 + this->formerCoreNum * this->formerCoreLength 
                        * this->addInput0Dim1234Length + (this->blockIdx - this->formerCoreNum) * this->tailCoreLength 
                        * this->addInput0Dim1234Length, this->tailCoreLength * this->addInput0Dim1234Length);
            }
            add0Input1Global.SetGlobalBuffer((__gm__ T *)add_0_input1, this->addInput0Dim14Length * sizeof(T));
            outGlobal.SetGlobalBuffer((__gm__ T *)out, (this->formerCoreNum * this->formerCoreLength 
                    + this->tailCoreNum * this->tailCoreLength) * this->addInput0Dim14Length);
            pipe.InitBuffer(inQueueAdd0Input0, 1, this->tileLength * this->addInput0Dim1234Length * sizeof(T));
            pipe.InitBuffer(inQueueAdd0Input1, 1, this->addInput0Dim14Length * sizeof(T));
            pipe.InitBuffer(inQueueMult1Input1, 1, this->tileLength * this->addInput0Dim23Length * sizeof(T));
            pipe.InitBuffer(inQueueMult2Input1, 1, this->tileLength * this->addInput0Dim1234Length * sizeof(T));
            pipe.InitBuffer(outQueue, 1, this->tileLength * this->addInput0Dim14Length * sizeof(T));
            pipe.InitBuffer(processBuf, this->tileLength * this->addInput0Dim1234Length * sizeof(T));
            pipe.InitBuffer(nBroadBuf, this->tileLength * this->addInput0Dim1234Length * sizeof(T));
        }

        __aicore__ inline void Process(GM_ADDR z, bool isNZ)
        {
            for (int32_t i = 0; i < this->tileNum; i++)
            {
                CopyIn(i);
                Compute(i, isNZ);
                CopyOut(i);
            }
        }

    private:
        __aicore__ inline void CopyIn(int32_t progress)
        {
            uint32_t ind = progress * this->tileLength;
            uint32_t length = this->tileLength;
            if (progress == this->tileNum - 1)
            {
                length = this->lastTileLength;
            }
            LocalTensor<T> add0Input0Local = inQueueAdd0Input0.AllocTensor<T>();
            LocalTensor<T> add0Input1Local = inQueueAdd0Input1.AllocTensor<T>();
            LocalTensor<T> mult1Input1Local = inQueueMult1Input1.AllocTensor<T>();
            LocalTensor<T> mult1Input2Local = inQueueMult2Input1.AllocTensor<T>();
            DataCopy(add0Input0Local, add0Input0Global[ind * this->addInput0Dim1234Length], length * this->addInput0Dim1234Length);
            DataCopy(add0Input1Local, add0Input1Global, this->addInput0Dim14Length);
            DataCopy(mult1Input1Local, mult1Input1Global[ind * this->addInput0Dim23Length], length * this->addInput0Dim23Length);
            DataCopy(mult1Input2Local, mult2Input1Global[ind * this->addInput0Dim1234Length], length * this->addInput0Dim1234Length);
            inQueueAdd0Input0.EnQue(add0Input0Local);
            inQueueAdd0Input1.EnQue(add0Input1Local);
            inQueueMult1Input1.EnQue(mult1Input1Local);
            inQueueMult2Input1.EnQue(mult1Input2Local);
        }

        __aicore__ inline void Compute(int32_t progress, bool isNZ)
        {
            uint32_t length = this->tileLength;
            if (progress == this->tileNum - 1)
            {
                length = this->lastTileLength;
            }
            LocalTensor<T> add0Input0Local = inQueueAdd0Input0.DeQue<T>();
            LocalTensor<T> add0Input1Local = inQueueAdd0Input1.DeQue<T>();
            LocalTensor<T> mult1Input1Local = inQueueMult1Input1.DeQue<T>();
            LocalTensor<T> mult1Input2Local = inQueueMult2Input1.DeQue<T>();

            LocalTensor<T> processLocal = processBuf.Get<T>();
            ComputeProcess(length, isNZ, add0Input0Local, add0Input1Local, 
                mult1Input1Local, mult1Input2Local, processLocal);
            
            PipeBarrier<PIPE_V>();
            LocalTensor<T> outLocal = outQueue.AllocTensor<T>();
            Duplicate<T>(outLocal, 0, this->tileLength * this->addInput0Dim14Length);
            if (isNZ) {
                for (int32_t i = 0; i < length; i++)
                {
                    for (int32_t j = 0; j < this->addInput0Dim1Length; j++)
                    {
                        Add(outLocal[i * addInput0Dim14Length + j * this->size_16], 
                        processLocal[i * this->addInput0Dim1234Length + j * this->addInput0Dim234Length], 
                        outLocal[i * this->addInput0Dim14Length + j * this->size_16], this->size_16, this->addInput0Dim23Length, {1, 1, 1, 0, 1, 0});
                    }
                }
            } else {
                for (int32_t i = 0; i < length; i++)
                {
                    Add(outLocal[i * this->addInput0Dim14Length], processLocal[i * this->addInput0Dim1234Length], 
                    outLocal[i * this->addInput0Dim14Length], this->addInput0Dim14Length, this->addInput0Dim23Length, {1, 1, 1, 0, this->blockOffset, 0});
                }
            }

            outQueue.EnQue(outLocal);
            inQueueAdd0Input0.FreeTensor(add0Input0Local);
            inQueueAdd0Input1.FreeTensor(add0Input1Local);
            inQueueMult1Input1.FreeTensor(mult1Input1Local);
            inQueueMult2Input1.FreeTensor(mult1Input2Local);
        }

        __aicore__ inline void ComputeProcess(uint32_t length, bool isNZ, LocalTensor<T> add0Input0Local,
            LocalTensor<T> add0Input1Local, LocalTensor<T> mult1Input1Local,
            LocalTensor<T> mult1Input2Local, LocalTensor<T> processLocal)
        {
            if (isNZ) {
                for (int32_t i = 0; i < length; i++)
                {
                    for (int32_t j = 0; j < this->addInput0Dim1Length; j++)
                    {
                        Add(processLocal[i * this->addInput0Dim1234Length + j * this->addInput0Dim234Length], 
                            add0Input0Local[i * this->addInput0Dim1234Length + j * this->addInput0Dim234Length], 
                            add0Input1Local[j * this->size_16], this->size_16, this->addInput0Dim23Length, {1, 1, 1, 1, 1, 0});
                    }
                }
            } else {
                Add(processLocal, add0Input0Local, add0Input1Local, this->addInput0Dim14Length, length * this->addInput0Dim23Length, {1, 1, 1, this->blockOffset, this->blockOffset, 0});
            }

            PipeBarrier<PIPE_V>();
            Muls(processLocal, processLocal, this->ignoreIndex, length * this->addInput0Dim1234Length);
            PipeBarrier<PIPE_V>();
            Sigmoid(processLocal, processLocal, length * this->addInput0Dim1234Length);

            if (!isNZ) {
                this->broadNum = this->addInput0Dim14Length;
            }

            LocalTensor<T> nBroadLocal = nBroadBuf.Get<T>();
            const int32_t broad_dim_2 = 2;
            uint32_t nShape[2] = {length * this->addInput0Dim23Length, 1};
            uint32_t nBroadShape[2] = {length * this->addInput0Dim23Length, this->broadNum};
            BroadCast<T, broad_dim_2, 1>(nBroadLocal, mult1Input1Local, nBroadShape, nShape);
            PipeBarrier<PIPE_V>();

            if (isNZ) {
                for (int32_t i = 0; i < length; i++)
                {
                    for (int32_t j = 0; j < this->addInput0Dim1Length; j++)
                    {
                        Mul(processLocal[i * this->addInput0Dim1234Length + j * this->addInput0Dim234Length], 
                        processLocal[i * this->addInput0Dim1234Length + j * this->addInput0Dim234Length], 
                        nBroadLocal[i * this->addInput0Dim234Length], this->addInput0Dim234Length);
                    }
                }
            } else {
                Mul(processLocal, processLocal, nBroadLocal, length * this->addInput0Dim1234Length);
            }
            
            PipeBarrier<PIPE_V>();
            Mul(processLocal, processLocal, mult1Input2Local, length * this->addInput0Dim1234Length);
        }

        __aicore__ inline void CopyOut(int32_t progress)
        {
            uint32_t ind = progress * this->tileLength;
            uint32_t length = this->tileLength;
            if (progress == this->tileNum - 1)
            {
                length = this->lastTileLength;
            }
            LocalTensor<T> outLocal = outQueue.DeQue<T>();
            DataCopy(outGlobal[this->outOffset + ind * this->addInput0Dim14Length], outLocal, length * this->addInput0Dim14Length);
            outQueue.FreeTensor(outLocal);
        }

    private:
        TPipe pipe;
        TQue<QuePosition::VECIN, 1> inQueueAdd0Input0;
        TQue<QuePosition::VECIN, 1> inQueueAdd0Input1;
        TQue<QuePosition::VECIN, 1> inQueueMul0Input1;
        TQue<QuePosition::VECIN, 1> inQueueMult1Input1;
        TQue<QuePosition::VECIN, 1> inQueueMult2Input1;
        TQue<QuePosition::VECOUT, 1> outQueue;
        TBuf<TPosition::VECCALC> processBuf;
        TBuf<TPosition::VECCALC> nBroadBuf;

        GlobalTensor<T> add0Input0Global;
        GlobalTensor<T> add0Input1Global;
        GlobalTensor<T> mul0Input1Global;
        GlobalTensor<T> mult1Input1Global;
        GlobalTensor<T> mult2Input1Global;
        GlobalTensor<T> outGlobal;

        int32_t blockIdx = 0;
        int32_t formerCoreNum;
        int32_t formerCoreLength;
        int32_t formerTileNum;
        int32_t formerTileLength;
        int32_t formerLastTileLength;
        int32_t tailCoreNum;
        int32_t tailCoreLength;
        int32_t tailTileNum;
        int32_t tailTileLength;
        int32_t tailLastTileLength;
        int32_t addInput0Dim1234Length;
        uint32_t addInput0Dim14Length;
        int32_t addInput0Dim23Length;
        int32_t addInput0Dim1Length;
        int32_t addInput0Dim234Length;
        T ignoreIndex;

        uint32_t tileNum;
        uint32_t tileLength;
        uint32_t lastTileLength;
        uint64_t outOffset = 0;
        uint8_t blockOffset = 0;
        uint32_t broadNum = 16;

        int32_t size_32 = 32;
        int32_t size_16 = 16;
    };
}

extern "C" __global__ __aicore__ void
add_sigmoid_mul_reduce_sum_d(GM_ADDR add_0_input0, GM_ADDR add_0_input1, GM_ADDR mul_0_input1, GM_ADDR mult_1_input1, GM_ADDR mult_2_input1, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace); 
    KernelAddSigmoidMulReduceSumD<float16_t> op;
    op.InitData(tiling_data.formerCoreNum,
        tiling_data.formerCoreLength,
        tiling_data.formerTileNum,
        tiling_data.formerTileLength,
        tiling_data.formerLastTileLength,
        tiling_data.tailCoreNum,
        tiling_data.tailCoreLength,
        tiling_data.tailTileNum,
        tiling_data.tailTileLength,
        tiling_data.tailLastTileLength,
        tiling_data.addInput0Dim1234Length,
        tiling_data.addInput0Dim14Length,
        tiling_data.addInput0Dim23Length,
        tiling_data.addInput0Dim1Length,
        tiling_data.addInput0Dim234Length);
    op.Init(add_0_input0, add_0_input1, mul_0_input1, mult_1_input1, mult_2_input1, out, usrWorkspace);
    if (TILING_KEY_IS(1))
    {
        op.Process(out, true);
    }
    else if (TILING_KEY_IS(2))
    {
        op.Process(out, false);
    }
}