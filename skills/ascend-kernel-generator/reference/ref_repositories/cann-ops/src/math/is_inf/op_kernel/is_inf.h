/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file is_inf.h
 * \brief
 */
#ifndef IS_INF_H
#define IS_INF_H

#include "kernel_operator.h"

namespace IsInfNS {

    using namespace AscendC;

    template <typename T, auto MASK, auto INF_MASK>
    class IsInf {
        public:
            __aicore__ inline IsInf() {};

            __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const IsInfTilingData* tilingData);
            __aicore__ inline void Process();

            constexpr static uint8_t BUFFER_NUM = 2;
            constexpr static uint8_t DATA_BLOCK = 32;
            constexpr static uint8_t FLOAT_INTERVAL_TYPE = 2;

        private:
            __aicore__ inline void CopyInX(int64_t offset, int32_t dataLength);
            __aicore__ inline void CopyOut(int32_t offset, int32_t dataLength);
            __aicore__ inline void ParseTilingData(const IsInfTilingData* tilingData);
            
            __aicore__ inline void ComputePerCore();

            __aicore__ inline void CompareInf(const int32_t dataLength);

            template <typename T1, typename T2>
            __aicore__ inline T1 CeilAlign(T1 a, T2 b) {
                return (a + b - 1) / b * b;
            };

        private:
            TPipe pipe;
            TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
            TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;

            TBuf<> cacheTensorBuff;
            GlobalTensor<int16_t> inputGM;
            GlobalTensor<uint8_t> outputGM;
            LocalTensor<int16_t> cacheTensor;

            uint8_t selectInterval = 0;
            uint32_t coreNum = 0;
            uint32_t tailCoreNum = 0;
            uint32_t perCoreDataCount = 0;
            uint32_t blockOffset = 0;
            uint32_t blockIdx = 0;
            uint32_t maxDataCount = 0;
            uint32_t actualMaxDataCount = 0;
            uint32_t usableUbSize = 0;
            uint32_t dataBlockSize = 0;
    };

    template <typename T, auto MASK, auto INF_MASK>
    __aicore__ inline void IsInf<T, MASK, INF_MASK>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const IsInfTilingData* tilingData) {
        
        inputGM.SetGlobalBuffer((__gm__ int16_t*)x);
        outputGM.SetGlobalBuffer((__gm__ uint8_t*)y);
        
        ParseTilingData(tilingData);

        maxDataCount = usableUbSize / sizeof(int16_t) / DATA_BLOCK * DATA_BLOCK;
        actualMaxDataCount  = maxDataCount / selectInterval;

        pipe.InitBuffer(inputQueue, BUFFER_NUM, maxDataCount * sizeof(int16_t));
        pipe.InitBuffer(outputQueue, BUFFER_NUM, actualMaxDataCount * sizeof(uint8_t));
        pipe.InitBuffer(cacheTensorBuff, maxDataCount * sizeof(int16_t));
    }

    template <typename T, auto MASK, auto INF_MASK>
    __aicore__ inline void IsInf<T, MASK, INF_MASK>::ParseTilingData(const IsInfTilingData* tilingData) {
        blockIdx = GetBlockIdx();
        coreNum = tilingData->needCoreNum;
        usableUbSize = tilingData->usableUbSize;
        perCoreDataCount = tilingData->perCoreDataCount;
        tailCoreNum = tilingData->tailDataCoreNum;

        selectInterval = sizeof(T) / sizeof(int16_t);

        if (tailCoreNum == 0) {
            blockOffset = perCoreDataCount * blockIdx;
        } else {
            if ((blockIdx + 1) <= tailCoreNum) {
                perCoreDataCount += DATA_BLOCK;
                blockOffset = perCoreDataCount * blockIdx;
            } else {
                blockOffset = ((perCoreDataCount + DATA_BLOCK) * tailCoreNum) + (perCoreDataCount * (blockIdx - tailCoreNum));
            }
        }

        if (blockIdx == coreNum - 1) {
            perCoreDataCount = tilingData->lastCoreDataCount;
        }

        blockOffset *= selectInterval;
        perCoreDataCount *= selectInterval;
    }

    template <typename T, auto MASK, auto INF_MASK>
    __aicore__ inline void IsInf<T, MASK, INF_MASK>::CopyInX(const int64_t offset, const int32_t dataLength) {

        LocalTensor<int16_t> dataLocal = inputQueue.AllocTensor<int16_t>();
        DataCopy(dataLocal, inputGM[offset], dataLength);
        
        inputQueue.EnQue(dataLocal);
    }

    template <typename T, auto MASK, auto INF_MASK>
    __aicore__ inline void IsInf<T, MASK, INF_MASK>::CopyOut(const int32_t offset, const int32_t dataLength) {

        LocalTensor<uint8_t> outLocal = outputQueue.DeQue<uint8_t>();

        DataCopy(outputGM[offset], outLocal, dataLength);
        outputQueue.FreeTensor(outLocal);
    }

    template <typename T, auto MASK, auto INF_MASK>
    __aicore__ inline void IsInf<T, MASK, INF_MASK>::CompareInf(const int32_t dataLength) {

        LocalTensor<int16_t> ubX = inputQueue.DeQue<int16_t>();
        LocalTensor<uint8_t> result = outputQueue.AllocTensor<uint8_t>();
        cacheTensor = cacheTensorBuff.Get<int16_t>();

        // 和sign_mask做按位与操作
        Duplicate(cacheTensor, (int16_t)MASK, dataLength);
        And(ubX, ubX, cacheTensor, dataLength);

        uint32_t actualCalCount = dataLength / selectInterval;
        if(selectInterval == FLOAT_INTERVAL_TYPE) {
            LocalTensor<int32_t> tmpInt32Tensor = ubX.ReinterpretCast<int32_t>();
            Adds(tmpInt32Tensor, tmpInt32Tensor, (int32_t)-INF_MASK, actualCalCount);
            And(ubX, ubX, cacheTensor, dataLength);
            Mins(tmpInt32Tensor, tmpInt32Tensor, (int32_t)1, actualCalCount);
            Muls(tmpInt32Tensor, tmpInt32Tensor, (int32_t)-1, actualCalCount);
            Adds(tmpInt32Tensor, tmpInt32Tensor, (int32_t)1, actualCalCount);
            Cast(ubX.ReinterpretCast<float>(), tmpInt32Tensor, RoundMode::CAST_NONE, actualCalCount);
            Cast(ubX.ReinterpretCast<half>(), ubX.ReinterpretCast<float>(), RoundMode::CAST_NONE, actualCalCount);
        } else {
            Adds(ubX, ubX, (int16_t)-INF_MASK, dataLength);
            And(ubX, ubX, cacheTensor, dataLength);
            Mins(ubX, ubX, (int16_t)1, dataLength);
            Muls(ubX, ubX, (int16_t)-1, dataLength);
            Adds(ubX, ubX, (int16_t)1, dataLength);
        }

        Cast(result, ubX.ReinterpretCast<half>(), RoundMode::CAST_CEIL, actualCalCount);

        inputQueue.FreeTensor(ubX);
        outputQueue.EnQue(result);
    }

    template <typename T, auto MASK, auto INF_MASK>
    __aicore__ inline void IsInf<T, MASK, INF_MASK>::Process() {
        if (blockIdx >= coreNum) {
            return;
        }

        ComputePerCore();
    }

    template <typename T, auto MASK, auto INF_MASK>
    __aicore__ inline void IsInf<T, MASK, INF_MASK>::ComputePerCore() {
        uint32_t loopCount = perCoreDataCount / maxDataCount;
        uint32_t tailDataCount = perCoreDataCount % maxDataCount;

        uint32_t actualInOffset = blockOffset;
        uint32_t actualOutOffset = blockOffset / selectInterval;
        uint32_t actualOutCount = maxDataCount / selectInterval;

        for (uint32_t idx = 0; idx < loopCount; idx ++) {
            CopyInX(actualInOffset, maxDataCount);
            CompareInf(maxDataCount);
            CopyOut(actualOutOffset, actualOutCount);
            actualOutOffset += actualOutCount;
            actualInOffset += maxDataCount;
        }

        if (tailDataCount > 0) {
            uint32_t dataBlock = DATA_BLOCK / sizeof(int16_t);
            uint32_t dataCount = CeilAlign(tailDataCount , dataBlock);
            actualOutCount = dataCount / selectInterval;
            actualOutCount = CeilAlign(actualOutCount , DATA_BLOCK);
            CopyInX(actualInOffset, dataCount);
            CompareInf(dataCount);
            CopyOut(actualOutOffset, actualOutCount);
        }
    }

}
#endif // IS_INF_H