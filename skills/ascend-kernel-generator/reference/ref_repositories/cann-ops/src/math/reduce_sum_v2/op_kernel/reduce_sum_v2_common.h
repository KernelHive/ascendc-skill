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
 * \file reduce_sum_v2_common.h
 * \brief
 */
#ifndef REDUCE_SUM_V2_COMMON_H
#define REDUCE_SUM_V2_COMMON_H

#include "kernel_operator.h"
#include "kernel_utils.h"

constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t REPEAT_SIZE = 256;
constexpr uint64_t REPEAT_STRIDE = 8;
constexpr uint64_t BUFFER_NUM = 2;
constexpr uint64_t CACHELINE = 512;
constexpr uint64_t WHOLE_REDUCE_SIZE = 256;
constexpr uint8_t NUM_2 = 2;
constexpr uint8_t NUM_3 = 3;
constexpr uint8_t NUM_4 = 4;
constexpr uint8_t CONTINOUS_PROCESS = 0;
constexpr uint8_t DISCONTINOUS_PROCESS = 1;
// Pattern
constexpr uint64_t PATTERN_AR = 0;
constexpr uint64_t PATTERN_ARA = 1;

enum class ReducePattern { AR, RA };

struct UbInfos {
    uint64_t formerATimes;
    uint64_t formerA;
    uint64_t tailA;
    uint64_t formerRTimes;
    uint64_t formerR;
    uint64_t tailR;
    uint64_t formerA1Times;
    uint64_t formerA1;
    uint64_t tailA1;
    uint64_t formerRealTimes;
    uint64_t tailRealData;
};

struct BlockInfos {
    uint64_t usedCoreNum;
    uint64_t formerCoreNum;
    uint64_t tailCoreNum;
    uint64_t formerUnitDataLen;
    uint64_t tailUnitDataLen;
    uint64_t tailRealDataLen;
};

namespace AscendC {
template<typename T_X, typename T_Y>
class ReduceSumV2KernelBase {
public:
    __aicore__ inline ReduceSumV2KernelBase() {}
    
protected:
    __aicore__ inline uint64_t FindCutPoint(const uint64_t &num)
    {
        uint64_t point = 1;
        while (point < num) {
            point <<= 1;
        }
        point >>= 1;
        return point;
    }

    __aicore__ inline void InitBlockParams(BlockInfos &blockInfos, const ReduceSumV2BlockInfos &blockTilingInfos)
    {
        blockInfos.usedCoreNum = blockTilingInfos.usedCoreNum;
        blockInfos.formerCoreNum = blockTilingInfos.formerCoreNum;
        blockInfos.tailCoreNum = blockTilingInfos.tailCoreNum;
        blockInfos.formerUnitDataLen = blockTilingInfos.formerUnitDataLen;
        blockInfos.tailUnitDataLen = blockTilingInfos.tailUnitDataLen;
        blockInfos.tailRealDataLen = blockTilingInfos.tailRealDataLen;
    }

    __aicore__ inline void InitBasicParams(const ReduceSumV2Process &tiling)
    {
        blockIdx_ = GetBlockIdx();
        xAlign_ = BLOCK_SIZE / sizeof(T_X);
        xFp32Align_ = BLOCK_SIZE / sizeof(float);
        uint64_t pattern = tiling.pattern;

        A = tiling.A;
        R = tiling.R;
        A1 = tiling.A1;

        for (size_t i = 0; i < NUM_2; i++) {
            ubInfos[i].formerATimes = tiling.ubInfos.formerATimes[i];
            ubInfos[i].formerA = tiling.ubInfos.formerA[i];
            ubInfos[i].tailA = tiling.ubInfos.tailA[i];
            ubInfos[i].formerRTimes = tiling.ubInfos.formerRTimes[i];
            ubInfos[i].formerR = tiling.ubInfos.formerR[i];
            ubInfos[i].tailR = tiling.ubInfos.tailR[i];
            ubInfos[i].formerA1Times = tiling.ubInfos.formerA1Times[i];
            ubInfos[i].formerA1 = tiling.ubInfos.formerA1[i];
            ubInfos[i].tailA1 = tiling.ubInfos.tailA1[i];
            ubInfos[i].formerRealTimes = tiling.ubInfos.formerRealTimes[i];
            ubInfos[i].tailRealData = tiling.ubInfos.tailRealData[i];
        }

        InitBlockParams(blockA, tiling.blockA);
        InitBlockParams(blockR, tiling.blockR);
    }

    __aicore__ inline uint64_t GetOffset(const uint64_t &blockIdx, const uint64_t &postDataNum, const BlockInfos &blockInfos)
    {
        uint64_t len;
        if (blockIdx < blockInfos.formerCoreNum) {
            len = blockIdx * blockInfos.formerUnitDataLen;
        } else {
            len = blockInfos.formerCoreNum * blockInfos.formerUnitDataLen +
                  (blockIdx - blockInfos.formerCoreNum) * blockInfos.tailUnitDataLen;
        }
        return len * postDataNum;
    }

    __aicore__ inline void MTE3ToMTE2Sync()
    {
        event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    }

    __aicore__ inline void VToMTE2Sync()
    {
        event_t eventIDVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
    }

    __aicore__ inline void MTE3ToVSync()
    {
        event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToV);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToV);
    }

    template <typename T>
    __aicore__ inline T Min(T a, T b)
    {
        return a > b ? b : a;
    }

    template <typename T>
    __aicore__ inline T Max(T a, T b)
    {
        return a < b ? b : a;
    }

protected:
    GlobalTensor<T_X> xGm_;
    GlobalTensor<T_Y> yGm_;

    uint64_t blockIdx_;
    uint64_t xAlign_;
    uint64_t xFp32Align_;
    uint64_t A;
    uint64_t R;
    uint64_t A1;
    UbInfos ubInfos[NUM_2];
    BlockInfos blockA;
    BlockInfos blockR;
    BlockInfos blockA1;
};
}

#endif // REDUCE_SUM_V2_COMMON_H
