/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file masked_select_v3.h
 * \brief
 */
#ifndef MASKED_SELECT_V3_H_
#define MASKED_SELECT_V3_H_

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
using namespace AscendC;
#define IS_1_BYTES_TYPE is_same<T, int8_t>::value || is_same<T, uint8_t>::value
#define IS_2_BYTES_TYPE is_same<T, int16_t>::value || is_same<T, uint16_t>::value || is_same<T, half>::value || is_same<T, bfloat16_t>::value
#define IS_4_BYTES_TYPE is_same<T, int32_t>::value || is_same<T, uint32_t>::value || is_same<T, float>::value
#define IS_8_BYTES_TYPE is_same<T, int64_t>::value || is_same<T, uint64_t>::value || is_same<T, double>::value

constexpr int32_t BUFFER_NUM = 1;

template <typename Tp, Tp v>
struct integral_constant {
    static constexpr Tp value = v;
};

using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;

template <typename, typename>
struct is_same : public false_type {};

template <typename Tp>
struct is_same<Tp, Tp> : public true_type {};
namespace AscendC {

constexpr uint32_t SHAPEOUT_SIZE = 2;
constexpr uint32_t BIT_NUM_PER_BYTE = 8;
constexpr uint32_t HEAD_BLOCK_SIZE = 64;
constexpr uint32_t OFFSET_SHIFT_BITS = 3; // offset偏移量移位输，<<3 等价于 *8
constexpr uint32_t INT64_LENGTH_IN_INT32 = 2; // INT64 相当于 2个int32长
constexpr uint32_t GATHER_RESULT_STRIDE = 8;

template <typename T>
class KernelMaskedSelectV3 {
public:
    __aicore__ inline KernelMaskedSelectV3 () {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR mask, GM_ADDR y, GM_ADDR shapeout, GM_ADDR workspace, 
                                uint32_t formerNum,
                                uint32_t formerLength,
                                uint32_t formertileNum,
                                uint32_t formertileLength,
                                uint32_t formerlasttileLength,
                                uint32_t tailNum,
                                uint32_t tailLength,
                                uint32_t tailtileNum,
                                uint32_t tailtileLength,
                                uint32_t taillasttileLength)
    {   
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockDim = GetBlockNum();
        __gm__ T* globalWorkTensor = (__gm__ T*)((__gm__ uint64_t*)workspace + this->blockDim * (HEAD_BLOCK_SIZE / sizeof(uint64_t)));

        blockIdx = GetBlockIdx();
        this->formerNum = formerNum;
        this->formerLength = formerLength;
        this->formertileNum = formertileNum;
        this->formertileLength = formertileLength;
        this->formerlasttileLength = formerlasttileLength;

        this->tailNum = tailNum;
        this->tailLength = tailLength;
        this->tailtileNum = tailtileNum;
        this->tailtileLength = tailtileLength;
        this->taillasttileLength = taillasttileLength;

        if (blockIdx < this->formerNum) {  //分到大块核的处理
            this->tileLength = this->formertileLength / BUFFER_NUM;
            this->lasttileLength = this->formerlasttileLength / BUFFER_NUM;
            this->tileNum = this->formertileNum * BUFFER_NUM;
            xGlobal.SetGlobalBuffer((__gm__ T*)x + this->formerLength * blockIdx, this->formerLength);
            maskGlobal.SetGlobalBuffer((__gm__ uint8_t*)mask + this->formerLength * blockIdx, this->formerLength);
            workGlobal.SetGlobalBuffer(globalWorkTensor + this->formerLength * blockIdx, this->formerLength);
        } else {  //分到小块核的处理，需要处理的数据量比大核少alignNum个
            this->tileLength = this->tailtileLength / BUFFER_NUM;
            this->lasttileLength = this->taillasttileLength / BUFFER_NUM;
            this->tileNum = this->tailtileNum * BUFFER_NUM;

            xGlobal.SetGlobalBuffer(
                (__gm__ T*)x + this->formerLength * this->formerNum +
                    this->tailLength * (blockIdx - this->formerNum),
                this->tailLength);
            maskGlobal.SetGlobalBuffer(
                (__gm__ uint8_t*)mask + this->formerLength * this->formerNum +
                    this->tailLength * (blockIdx - this->formerNum),
                this->tailLength);
            workGlobal.SetGlobalBuffer(
                globalWorkTensor + this->formerLength * this->formerNum +
                    this->tailLength * (blockIdx - this->formerNum),
                this->tailLength);
        }
        shapeoutGlobal.SetGlobalBuffer((__gm__ uint64_t*)shapeout, SHAPEOUT_SIZE);
        offsetGlobal.SetGlobalBuffer((__gm__ uint64_t*)workspace, blockDim);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(inQueueMask, BUFFER_NUM, this->tileLength * sizeof(uint8_t));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(moveQue, BUFFER_NUM, this->tileLength * sizeof(T));

        if constexpr (IS_8_BYTES_TYPE) {
            pipe.InitBuffer(maskCastBuf, this->tileLength * sizeof(float));
            pipe.InitBuffer(bitMaskBuf, this->tileLength * INT64_LENGTH_IN_INT32 / BIT_NUM_PER_BYTE);
        } else {
            pipe.InitBuffer(maskCastBuf, this->tileLength * sizeof(half));
            pipe.InitBuffer(bitMaskBuf, this->tileLength / BIT_NUM_PER_BYTE);
        }

        if constexpr (IS_1_BYTES_TYPE) {
            pipe.InitBuffer(xCastBuf, this->tileLength * sizeof(half));
            pipe.InitBuffer(yCastBuf, this->tileLength * sizeof(half));
        }
    }

    __aicore__ inline void Process(GM_ADDR y, GM_ADDR shapeout)
    {
        int32_t loopCount = this->tileNum ;
        //GYW 先处理可以整分的。
        for (int32_t i = 0; i < loopCount; ++i) {
            CopyIn(i);
            Compute(i);
            CopyOut2WorkSpace();
        }
        //workspace 写入 offset
        offsetGlobal.SetValue(blockIdx<<OFFSET_SHIFT_BITS, this->outOffset);
        DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(offsetGlobal[blockIdx<<OFFSET_SHIFT_BITS]);
        SyncAll();
        uint64_t ind = 0;
        for (int32_t i = 0; i < blockIdx; i++) { 
            DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(offsetGlobal[i<<OFFSET_SHIFT_BITS]);
            ind += offsetGlobal.GetValue(i<<OFFSET_SHIFT_BITS);
        }
        
        yGlobal.SetGlobalBuffer((__gm__ T*)y + ind, this->outOffset);
        
        //搬运至GM
        loopCount = this->outOffset / this->tileLength;
        int32_t tailLoopLength = this->outOffset % this->tileLength;
        //GYW 先处理可以整分的。
        for (int32_t i = 0; i < loopCount; ++i) {
            CopyInMove(i, this->tileLength);
            CopyOutMove(i, this->tileLength);
        }
        //剩余不能被整分处理
        if (tailLoopLength > 0) {
            CopyInMove(loopCount, tailLoopLength);
            CopyOutMove(loopCount, tailLoopLength);
        }
        if (this->blockIdx == this->blockDim -1) {
            shapeoutGlobal.SetValue(0, 1);
            shapeoutGlobal.SetValue(1, ind + this->outOffset);
        }
}
private:
    __aicore__ inline void CopyInMove(int32_t progress, int32_t length)
    {
        LocalTensor<T> xLocal = moveQue.AllocTensor<T>();
        
        if constexpr (IS_8_BYTES_TYPE) {//int64 uint64 double
            DataCopyPadDoubleWord(xLocal, workGlobal[progress * (this->tileLength)], length);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal, workGlobal[progress * (this->tileLength)], copyParams, padParams);
        }
        moveQue.EnQue(xLocal);
    }

    __aicore__ inline void CopyOutMove(int32_t progress,int32_t length)
    {
        LocalTensor<T> yLocal = moveQue.DeQue<T>();
        
        if constexpr (IS_8_BYTES_TYPE) {
            DataCopyPadDoubleWord(yGlobal[progress * (this->tileLength)], yLocal, length);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
            DataCopyPad(yGlobal[progress * (this->tileLength)], yLocal, copyParams);
        }
        moveQue.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        LocalTensor<uint8_t> maskLocal = inQueueMask.AllocTensor<uint8_t>();
        uint32_t ind = progress * this->tileLength;
        uint32_t length = this->tileLength;
        if (progress == this->tileNum - 1) {
            // 最后一个block，最后一个tile
            length = this->lasttileLength;
        } 

        if constexpr (IS_8_BYTES_TYPE) {//int64 uint64 double
            DataCopyPadDoubleWord(xLocal, xGlobal[ind], length);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal, xGlobal[ind], copyParams, padParams);
        }

        {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(length), 0, 0, 0};
            DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
            DataCopyPad(maskLocal, maskGlobal[ind], copyParams, padParams);
        }

        inQueueX.EnQue(xLocal);
        inQueueMask.EnQue(maskLocal);
    }

__aicore__ inline void GenerateMask(const LocalTensor<uint8_t>& mask, LocalTensor<uint8_t>& bitMask,uint32_t count)
    {
        LocalTensor<half> maskCastLocal = maskCastBuf.Get<half>();

        Duplicate(maskCastLocal, static_cast<half>(0), static_cast<int32_t>(this->tileLength));
        Cast(maskCastLocal, mask, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();

        if constexpr (IS_8_BYTES_TYPE) {
            LocalTensor<int16_t> maskCastInt16 = maskCastLocal.template ReinterpretCast<int16_t>();
            LocalTensor<int16_t> maskCastInt16Shift = maskCastLocal[this->tileLength].template ReinterpretCast<int16_t>();
            Cast(maskCastInt16, maskCastLocal, RoundMode::CAST_ROUND, this->tileLength);

            ShiftLeft(maskCastInt16Shift, maskCastInt16, static_cast<int16_t>(BIT_NUM_PER_BYTE), this->tileLength);
            Add(maskCastInt16Shift, maskCastInt16, maskCastInt16Shift, this->tileLength);

            Cast(maskCastLocal, maskCastInt16Shift.ReinterpretCast<uint8_t>(), RoundMode::CAST_NONE, this->tileLength * INT64_LENGTH_IN_INT32);
            CompareScalar(bitMask, maskCastLocal, static_cast<half>(1.0), CMPMODE::EQ, this->tileLength * INT64_LENGTH_IN_INT32);
        } else {
            CompareScalar(bitMask, maskCastLocal, static_cast<half>(1.0), CMPMODE::EQ, this->tileLength);
        }
    }

__aicore__ inline void GatherResult(LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
                                        const LocalTensor<uint8_t>& bitMaskLocal, int32_t count)
    {
        GatherMaskParams params;
        params.src0BlockStride = 1;
        params.repeatTimes = 1;
        params.src0RepeatStride = GATHER_RESULT_STRIDE;
        params.src1RepeatStride = 1;

        if constexpr (IS_8_BYTES_TYPE) {
            uint32_t mask = count * INT64_LENGTH_IN_INT32;
            LocalTensor<uint32_t> bitMask = bitMaskLocal.ReinterpretCast<uint32_t>();
            LocalTensor<int32_t> dstCastLocal = dstLocal.template ReinterpretCast<int32_t>();
            LocalTensor<int32_t> srcCastLocal = srcLocal.template ReinterpretCast<int32_t>();
            GatherMask(dstCastLocal, srcCastLocal, bitMask, true, mask, params, rsvdCnt);
        } else if constexpr (IS_4_BYTES_TYPE) {
            uint32_t mask = count;
            LocalTensor<uint32_t> bitMask = bitMaskLocal.ReinterpretCast<uint32_t>();
            GatherMask(dstLocal, srcLocal, bitMask, true, mask, params, rsvdCnt);
        } else if constexpr (IS_2_BYTES_TYPE) {
            uint32_t mask = count;
            LocalTensor<uint16_t> bitMask = bitMaskLocal.ReinterpretCast<uint16_t>();
            GatherMask(dstLocal, srcLocal, bitMask, true, mask, params, rsvdCnt);//rsvdCnt 最终有效元素个数
        } else {
            uint32_t mask = count;
            LocalTensor<half> xCastLocal = xCastBuf.Get<half>();
            LocalTensor<half> yCastLocal = yCastBuf.Get<half>();
            Duplicate(xCastLocal, static_cast<half>(0), static_cast<int32_t>(this->tileLength));
            Cast(xCastLocal, srcLocal, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            LocalTensor<uint16_t> bitMask = bitMaskLocal.ReinterpretCast<uint16_t>();
            GatherMask(yCastLocal, xCastLocal, bitMask, true, mask, params, rsvdCnt);
            Cast(dstLocal, yCastLocal, RoundMode::CAST_NONE, rsvdCnt);
        }
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<uint8_t> maskLocal = inQueueMask.DeQue<uint8_t>();
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        LocalTensor<uint8_t> bitMaskLocal = bitMaskBuf.Get<uint8_t>();// GYW  DeQue 和 GET区别？

        uint32_t length = this->tileLength;
        if (progress == this->tileNum - 1) {
            length = this->lasttileLength;
        }
        GenerateMask(maskLocal, bitMaskLocal, length);
        GatherResult(yLocal, xLocal, bitMaskLocal, length);
        
        outQueueY.EnQue<T>(yLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueMask.FreeTensor(maskLocal);
    }

    __aicore__ inline void DataCopyPadDoubleWord(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal,
                                                int64_t count)
    {
        GlobalTensor<int32_t> srcCastGlobal;
        srcCastGlobal.SetGlobalBuffer((__gm__ int32_t*)srcGlobal.GetPhyAddr(), count * INT64_LENGTH_IN_INT32);//将GM 中 64 转成 32 * 2

        LocalTensor<int32_t> dstCastLocal = dstLocal.template ReinterpretCast<int32_t>();//将 ue转 int32

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * INT64_LENGTH_IN_INT32 * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(dstCastLocal, srcCastGlobal, copyParams, padParams); 
    }

    __aicore__ inline void DataCopyPadDoubleWord(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal,
                                                int64_t count)
    {
        GlobalTensor<int32_t> dstCastGlobal;
        dstCastGlobal.SetGlobalBuffer((__gm__ int32_t*)dstGlobal.GetPhyAddr(), count * INT64_LENGTH_IN_INT32);

        LocalTensor<int32_t> srcCastLocal = srcLocal.template ReinterpretCast<int32_t>();

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * INT64_LENGTH_IN_INT32 * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(dstCastGlobal, srcCastLocal, copyParams);
    }

    __aicore__ inline void CopyOut2WorkSpace()
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();

        if constexpr (IS_8_BYTES_TYPE) {
            DataCopyPadDoubleWord(workGlobal[outOffset], yLocal, rsvdCnt / INT64_LENGTH_IN_INT32);
            outOffset += rsvdCnt / INT64_LENGTH_IN_INT32;
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(rsvdCnt * sizeof(T)), 0, 0, 0};
            DataCopyPad(workGlobal[outOffset], yLocal, copyParams);
            outOffset += rsvdCnt;
        }

        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueMask;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueDst;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> moveQue;
    TBuf<TPosition::VECCALC> maskCastBuf;
    TBuf<TPosition::VECCALC> bitMaskBuf;
    TBuf<TPosition::VECCALC> xCastBuf;
    TBuf<TPosition::VECCALC> yCastBuf;

    GlobalTensor<T> xGlobal;
    GlobalTensor<T> yGlobal;
    GlobalTensor<uint64_t> shapeoutGlobal;
    GlobalTensor<uint8_t> maskGlobal;
    GlobalTensor<T> workGlobal;
    GlobalTensor<uint64_t> offsetGlobal;

    // 输入
    uint32_t blockDim;
    uint32_t formerNum;
    uint32_t formerLength;
    uint32_t formertileNum;
    uint32_t formertileLength;
    uint32_t formerlasttileLength;
    uint32_t tailNum;
    uint32_t tailLength;
    uint32_t tailtileNum;
    uint32_t tailtileLength;
    uint32_t taillasttileLength;

    // 本block/核的
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t lasttileLength;

    uint64_t rsvdCnt = 0;
    uint64_t outOffset = 0;
    uint32_t blockIdx = 0;
};
} // namespace AscendC

#endif // MASKED_SELECT_V3_H_
