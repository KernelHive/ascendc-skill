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
 * @file fill.cpp
 */
#include "kernel_operator.h"

constexpr uint32_t BUFFER_NUM = 1;

template <typename T, bool IsExistBigCore>
class KernelFill {
   public:
    __aicore__ inline KernelFill() {
    }
    __aicore__ inline void Init(GM_ADDR dims, GM_ADDR value, GM_ADDR y, uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, uint32_t bigCoreLoopNum, uint32_t smallCoreLoopNum,
                                uint32_t ubPartDataNum, uint32_t smallCoreTailDataNum, uint32_t bigCoreTailDataNum,
                                uint32_t tailBlockNum) {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t coreNum = AscendC::GetBlockIdx();

        uint32_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();
        this->ubPartDataNum = ubPartDataNum;

        if constexpr (IsExistBigCore) {
            if (coreNum < tailBlockNum) {
                this->coreDataNum = bigCoreDataNum;
                this->tileNum = bigCoreLoopNum;
                this->tailDataNum = bigCoreTailDataNum;
            } else {
                this->coreDataNum = smallCoreDataNum;
                this->tileNum = smallCoreLoopNum;
                this->tailDataNum = smallCoreTailDataNum;
                globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (AscendC::GetBlockIdx() - tailBlockNum);
            }
        } else {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = smallCoreLoopNum;
            this->tailDataNum = smallCoreTailDataNum;
            globalBufferIndex = smallCoreDataNum * AscendC::GetBlockIdx();
        }

        this->value = *reinterpret_cast<__gm__ DTYPE_VALUE *>(value);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + globalBufferIndex, this->coreDataNum);

        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, bool>) {
            pipe.InitBuffer(tmpBuf, this->ubPartDataNum * sizeof(half));
        }

        pipe.InitBuffer(outQueueOUT, BUFFER_NUM, this->ubPartDataNum * sizeof(DTYPE_Y));
    }
    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->ubPartDataNum;
        for (int32_t i = 0; i < loopCount - 1; i++) {
            Compute(i);
            CopyOut(i);
        }
        this->processDataNum = this->tailDataNum;
        Compute(loopCount - 1);
        CopyOut(loopCount - 1);
    }

   private:
    __aicore__ inline void Compute(uint32_t progress) {
        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, bool>) {
            AscendC::LocalTensor<int8_t> outLocal = outQueueOUT.AllocTensor<int8_t>();
            AscendC::LocalTensor<half> tmpLocal = tmpBuf.Get<half>();
            AscendC::Duplicate<half>(tmpLocal, (half)(this->value), this->processDataNum);
            Cast(outLocal, tmpLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            outQueueOUT.EnQue<int8_t>(outLocal);
        } else {
            AscendC::LocalTensor<T> outLocal = outQueueOUT.AllocTensor<T>();
            AscendC::Duplicate<T>(outLocal, this->value, this->processDataNum);
            outQueueOUT.EnQue<T>(outLocal);
        }
    }

    __aicore__ inline void CopyOut(uint32_t progress) {
        AscendC::LocalTensor<DTYPE_VALUE> outLocal = outQueueOUT.DeQue<DTYPE_VALUE>();
        AscendC::DataCopy(yGm[progress * this->ubPartDataNum], outLocal, this->processDataNum);
        outQueueOUT.FreeTensor(outLocal);
    }

   private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueOUT;
    AscendC::GlobalTensor<DTYPE_Y> yGm;

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    T value;

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t ubPartDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

template <bool IsExistBigCore>
class KernelFill1_INT64 {
   public:
    __aicore__ inline KernelFill1_INT64() {
    }
    __aicore__ inline void Init(GM_ADDR dims, GM_ADDR values, GM_ADDR y, uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, uint32_t bigCoreLoopNum, uint32_t smallCoreLoopNum,
                                uint32_t ubPartDataNum, uint32_t smallCoreTailDataNum, uint32_t bigCoreTailDataNum,
                                uint32_t tailBlockNum) {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t coreNum = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();
        this->ubPartDataNum = ubPartDataNum;

        if constexpr (IsExistBigCore) {
            if (coreNum < tailBlockNum) {
                this->coreDataNum = bigCoreDataNum;
                this->tileNum = bigCoreLoopNum;
                this->tailDataNum = bigCoreTailDataNum;
            } else {
                this->coreDataNum = smallCoreDataNum;
                this->tileNum = smallCoreLoopNum;
                this->tailDataNum = smallCoreTailDataNum;
                globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (AscendC::GetBlockIdx() - tailBlockNum);
            }
        } else {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = smallCoreLoopNum;
            this->tailDataNum = smallCoreTailDataNum;
            globalBufferIndex = smallCoreDataNum * AscendC::GetBlockIdx();
        }

        xGm.SetGlobalBuffer((__gm__ int32_t *)values, 2);
        yGm.SetGlobalBuffer((__gm__ int32_t *)y + globalBufferIndex, this->coreDataNum);

        this->high = xGm.GetValue(1);
        this->low = xGm.GetValue(0);

        pipe.InitBuffer(outQueueOUT, BUFFER_NUM, this->ubPartDataNum * sizeof(int32_t));
    }
    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->ubPartDataNum;
        this->repeatTimes = (this->processDataNum + 63) / 64;
        for (int32_t i = 0; i < loopCount - 1; i++) {
            Compute(i);
            CopyOut(i);
        }
        this->processDataNum = this->tailDataNum;
        this->repeatTimes = (this->processDataNum + 63) / 64;
        Compute(loopCount - 1);
        CopyOut(loopCount - 1);
    }

   private:
    __aicore__ inline void Compute(uint32_t progress) {
        AscendC::LocalTensor<int32_t> outLocal = outQueueOUT.AllocTensor<int32_t>();

        uint64_t mask2[2] = {0xAAAAAAAAAAAAAAAA, 0};
        uint64_t mask1[2] = {0x5555555555555555, 0};

        AscendC::Duplicate(outLocal, low, mask1, this->repeatTimes, 1, 8);
        AscendC::Duplicate(outLocal, high, mask2, this->repeatTimes, 1, 8);

        outQueueOUT.EnQue<int32_t>(outLocal);
    }
    __aicore__ inline void CopyOut(uint32_t progress) {
        AscendC::LocalTensor<int32_t> outLocal = outQueueOUT.DeQue<int32_t>();

        AscendC::DataCopy(yGm[progress * this->ubPartDataNum], outLocal, this->processDataNum);

        outQueueOUT.FreeTensor(outLocal);
    }

   private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueOUT;
    AscendC::GlobalTensor<int32_t> yGm;
    AscendC::GlobalTensor<int32_t> xGm;

    int64_t value;

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t ubPartDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;

    int32_t high;
    int32_t low;
    uint32_t repeatTimes;
};
extern "C" __global__ __aicore__ void fill(GM_ADDR dims, GM_ADDR value, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    if (TILING_KEY_IS(0)) {
        if constexpr (std::is_same_v<DTYPE_VALUE, bool> || std::is_same_v<DTYPE_VALUE, int8_t>) {
            KernelFill<int8_t, false> op;
            op.Init(dims, value, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum,
                    tiling_data.bigCoreLoopNum, tiling_data.smallCoreLoopNum, tiling_data.ubPartDataNum,
                    tiling_data.smallCoreTailDataNum, tiling_data.bigCoreTailDataNum, tiling_data.tailBlockNum);
            op.Process();
        } else if constexpr (std::is_same_v<DTYPE_VALUE, int64_t>) {
            KernelFill1_INT64<false> op;
            op.Init(dims, value, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum,
                    tiling_data.bigCoreLoopNum, tiling_data.smallCoreLoopNum, tiling_data.ubPartDataNum,
                    tiling_data.smallCoreTailDataNum, tiling_data.bigCoreTailDataNum, tiling_data.tailBlockNum);
            op.Process();
        } else {
            KernelFill<DTYPE_VALUE, false> op;
            op.Init(dims, value, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum,
                    tiling_data.bigCoreLoopNum, tiling_data.smallCoreLoopNum, tiling_data.ubPartDataNum,
                    tiling_data.smallCoreTailDataNum, tiling_data.bigCoreTailDataNum, tiling_data.tailBlockNum);
            op.Process();
        }
    } else if (TILING_KEY_IS(1)) {
        if constexpr (std::is_same_v<DTYPE_VALUE, bool> || std::is_same_v<DTYPE_VALUE, int8_t>) {
            KernelFill<int8_t, true> op;
            op.Init(dims, value, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum,
                    tiling_data.bigCoreLoopNum, tiling_data.smallCoreLoopNum, tiling_data.ubPartDataNum,
                    tiling_data.smallCoreTailDataNum, tiling_data.bigCoreTailDataNum, tiling_data.tailBlockNum);
            op.Process();
        } else if constexpr (std::is_same_v<DTYPE_VALUE, int64_t>) {
            KernelFill1_INT64<true> op;
            op.Init(dims, value, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum,
                    tiling_data.bigCoreLoopNum, tiling_data.smallCoreLoopNum, tiling_data.ubPartDataNum,
                    tiling_data.smallCoreTailDataNum, tiling_data.bigCoreTailDataNum, tiling_data.tailBlockNum);
            op.Process();
        } else {
            KernelFill<DTYPE_VALUE, true> op;
            op.Init(dims, value, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum,
                    tiling_data.bigCoreLoopNum, tiling_data.smallCoreLoopNum, tiling_data.ubPartDataNum,
                    tiling_data.smallCoreTailDataNum, tiling_data.bigCoreTailDataNum, tiling_data.tailBlockNum);
            op.Process();
        }
    }
}