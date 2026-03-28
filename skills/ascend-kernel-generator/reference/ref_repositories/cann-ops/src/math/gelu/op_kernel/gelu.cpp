/** 
 * @file gelu.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class KernelGelu
{
public:
    __aicore__ inline KernelGelu() {}
    __aicore__ inline void Init(
        GM_ADDR src_gm, 
        GM_ADDR dst_gm,
        uint32_t totalLength,
        uint32_t ALIGN_NUM,
        uint32_t block_size,
        uint32_t core_size,
        uint32_t core_remain
        )
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero");
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);
        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);
        auto startPointer = core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;
        src_global.SetGlobalBuffer((__gm__ T *)src_gm + startPointer, bufferlength);
        dst_global.SetGlobalBuffer((__gm__ T *)dst_gm + startPointer, bufferlength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(T));
        //分出两块TBuffer存放计算过程中产生的临时Tensor
        if constexpr ( ! std::is_same_v<T, float32_t>)
            pipe.InitBuffer(tmp1, this->tileLength * sizeof(float));
            pipe.InitBuffer(tmp2, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount - 1; i++)
        {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }
        auto length = this->blockLength - this->tileLength * (loopCount - 1);
        CopyIn(loopCount-1, length);
        Compute(loopCount-1, length);
        CopyOut(loopCount-1, length);
    }

private:
    __aicore__ inline void CopyIn(uint32_t process, uint32_t length)
    {
        LocalTensor<T> srcLocal = inQueueX.AllocTensor<T>();
        DataCopy(srcLocal, src_global[process * this->tileLength], length);
        inQueueX.EnQue(srcLocal);
    }
    __aicore__ inline void Compute(uint32_t process, uint32_t length)
    {
        LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
        LocalTensor<T> srcLocal = inQueueX.DeQue<T>();
        //计算前加入输入tensor对float32的转换，计算后转回原类型
        if constexpr ( ! std::is_same_v<T, float32_t>)
        {
            LocalTensor<float> p1 = tmp1.Get<float>();
            LocalTensor<float> p2 = tmp2.Get<float>();
            Cast(p1, srcLocal, RoundMode::CAST_NONE, length);
            Cast(p2, srcLocal, RoundMode::CAST_NONE, length);
            Mul(p2, p1, p1, length);
            Mul(p2, p2, p1, length);
            Muls(p2, p2, (float)this->param1, length);
            Add(p2, p2, p1, length);
            Muls(p2, p2, (float)this->param2, length);
            Exp(p2,p2,length);
            Adds(p2, p2, (float)1, length);
            Div(p2,p1,p2,length);            
            Cast(dstLocal, p2, RoundMode::CAST_RINT, length);
	    tmp1.FreeTensor(p1);
	    tmp2.FreeTensor(p2);
        }
        else
        {
            Mul(dstLocal, srcLocal, srcLocal, length);
            Mul(dstLocal, dstLocal, srcLocal, length);
            Muls(dstLocal, dstLocal, (T)this->param1, length);
            Add(dstLocal, dstLocal, srcLocal, length);
            Muls(dstLocal, dstLocal, (T)this->param2, length);
            Exp(dstLocal,dstLocal,length);
            Adds(dstLocal, dstLocal, (T)1, length);
            Div(dstLocal,srcLocal,dstLocal,length);
        }    
        outQueue.EnQue<T>(dstLocal);
        inQueueX.FreeTensor(srcLocal);
    }

    __aicore__ inline void CopyOut(uint32_t process, uint32_t length)
    {
        LocalTensor<T> dstLocal = outQueue.DeQue<T>();
        DataCopy(dst_global[process * this->tileLength], dstLocal, length);
        outQueue.FreeTensor(dstLocal);
    }

private:
    GlobalTensor<T> src_global;
    GlobalTensor<T> dst_global;
    TBuf<QuePosition::VECCALC> tmp1;
    TBuf<QuePosition::VECCALC> tmp2;
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    float param1 = 0.0455399241;
    float param2 = -1.595769122;
};


extern "C" __global__ __aicore__ void gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelGelu<DTYPE_X> op;
    op.Init(x, y,
            tiling_data.totalLength,
            tiling_data.ALIGN_NUM,
            tiling_data.block_size,
            tiling_data.core_size,
            tiling_data.core_remain);
    op.Process();
}
