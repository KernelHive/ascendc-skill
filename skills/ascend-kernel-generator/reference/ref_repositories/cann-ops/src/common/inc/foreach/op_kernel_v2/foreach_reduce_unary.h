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
 * \file kernel_foreach_reduce_one_scalar_binary.h
 * \brief
 */

#ifndef KERNEL_FOREACH_REDUCE_ONE_SCALAR_BINARY_H
#define KERNEL_FOREACH_REDUCE_ONE_SCALAR_BINARY_H

#include "kernel_foreach_reduce.h"

namespace Common {
namespace OpKernel {
using namespace AscendC;

template <typename P, typename OpPredicate>
class ReduceAdapter {
public:
    explicit __aicore__ inline ReduceAdapter(OpPredicate &p): pred(p) {};
    __aicore__ inline void BeforeReduceOp(const LocalTensor<P> &dstLocal, const LocalTensor<P> &srcLocal, int64_t dataCount) {
        static_assert(std::is_member_function_pointer_v<decltype(&OpPredicate::BeforeReduceOp)>);
        pred.BeforeReduceOp(dstLocal, srcLocal, dataCount);
    }

    __aicore__ inline void AfterReduceOp(const LocalTensor<P> &dstLocal, const LocalTensor<P> &srcLocal, int64_t dataCount) {
        static_assert(std::is_member_function_pointer_v<decltype(&OpPredicate::AfterReduceOp)>);
        pred.AfterReduceOp(dstLocal, srcLocal, dataCount);
    }

    __aicore__ inline void ReduceOp(const LocalTensor<P>& dstLocal, const LocalTensor<P>& srcLocal, const LocalTensor<P>& workLocal, int32_t count) {
        static_assert(std::is_member_function_pointer_v<decltype(&OpPredicate::ReduceOp)>);
        pred.ReduceOp(dstLocal, srcLocal, workLocal, count);
    }
private:
    OpPredicate &pred;
};

template<typename T, typename P, typename OpPredicate>
class InnerComputer {
private:
    __aicore__ inline void Round1ComputePerCast(ReduceAdapter<P, OpPredicate> &reduceAdapter,
        LocalTensor<T> &dataLocal, LocalTensor<float> &float32Tensor,
        uint32_t maxCastDataCount, uint16_t index, int64_t dataCount) {
        pipe_barrier(PIPE_V);
        Cast(float32Tensor, dataLocal[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
        pipe_barrier(PIPE_V);
        reduceAdapter.BeforeReduceOp(float32Tensor, float32Tensor, dataCount);
        pipe_barrier(PIPE_V);
        reduceAdapter.ReduceOp(float32Tensor, float32Tensor, float32Tensor, dataCount);

        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        SetValueAdapter<float>(float32Tensor, float32Tensor.GetValue(0), maxCastDataCount + index);
        SetFlag<HardEvent::S_V>(EVENT_ID1);
        WaitFlag<HardEvent::S_V>(EVENT_ID1);
    }
public:
    __aicore__ inline void ComputeRound1(ReduceAdapter<P, OpPredicate> &reduceAdapter,
        LocalTensor<T> &dataLocal, LocalTensor<P> &tempLocal, LocalTensor<float> &float32Tensor,
        uint32_t maxCastDataCount, int64_t dataCount, uint16_t tempIndex) {
        uint32_t castTimes = 0;
        uint32_t castDatacountRemainder = 0;
        if (maxCastDataCount == 0) {
            castTimes = -1;
            castDatacountRemainder = -1;
        } else {
            castTimes = dataCount / maxCastDataCount;
            castDatacountRemainder = dataCount % maxCastDataCount;
        }

        for (uint32_t i = 0; i < castTimes; i++) {
            Round1ComputePerCast(
            reduceAdapter, dataLocal, float32Tensor, maxCastDataCount, i, maxCastDataCount);
        }
        if (castDatacountRemainder > 0) {
            Round1ComputePerCast(
            reduceAdapter, dataLocal, float32Tensor, maxCastDataCount, castTimes, castDatacountRemainder);
            castTimes++;
        }
        pipe_barrier(PIPE_V);
        reduceAdapter.ReduceOp(float32Tensor, float32Tensor[maxCastDataCount], float32Tensor[maxCastDataCount], castTimes);

        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        SetValueAdapter<P>(tempLocal, float32Tensor.GetValue(0), tempIndex);
        SetFlag<HardEvent::S_V>(EVENT_ID1);
        WaitFlag<HardEvent::S_V>(EVENT_ID1);
    }

    __aicore__ inline void ComputeRound2(ReduceAdapter<P, OpPredicate> &reduceAdapter,
        LocalTensor<P> &dataLocal, LocalTensor<T> &outLocal, int64_t dataCount) {
        if (dataCount > 1) {
            pipe_barrier(PIPE_V);
            reduceAdapter.ReduceOp(dataLocal, dataLocal, dataLocal, dataCount);
        }
        pipe_barrier(PIPE_V);
        reduceAdapter.AfterReduceOp(dataLocal, dataLocal, 1);
        pipe_barrier(PIPE_V);
        Cast(outLocal, dataLocal, RoundMode::CAST_RINT, 1);
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void ReduceCompute(ReduceAdapter<P, OpPredicate> &reduceAdapter,
        LocalTensor<P> &dataLocal, LocalTensor<P> &outLocal, LocalTensor<P>& workLocal, int64_t dataCount) {
        pipe_barrier(PIPE_V);
        reduceAdapter.ReduceOp(dataLocal, outLocal, workLocal, dataCount);
        pipe_barrier(PIPE_V);
    }
};

template<typename P, typename OpPredicate>
class InnerComputer<float, P, OpPredicate> {
public:
    __aicore__ inline void ComputeRound1(ReduceAdapter<P, OpPredicate> &reduceAdapter,
        LocalTensor<float> &dataLocal, LocalTensor<P> &tempLocal, LocalTensor<float> &float32Tensor,
        uint32_t maxCastDataCount, int64_t dataCount, uint16_t tempIndex) {
        pipe_barrier(PIPE_V);
        reduceAdapter.BeforeReduceOp(dataLocal, dataLocal, dataCount);
        pipe_barrier(PIPE_V);
        reduceAdapter.ReduceOp(dataLocal, dataLocal, dataLocal, dataCount);
        pipe_barrier(PIPE_V);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        SetValueAdapter<P>(tempLocal, dataLocal.GetValue(0), tempIndex);
        SetFlag<HardEvent::S_V>(EVENT_ID1);
        WaitFlag<HardEvent::S_V>(EVENT_ID1);
    }

    __aicore__ inline void ComputeRound2(ReduceAdapter<P, OpPredicate> &reduceAdapter,
        LocalTensor<float> &dataLocal, LocalTensor<float> &outLocal, int64_t dataCount) {
        if (dataCount > 1) {
            pipe_barrier(PIPE_V);
            reduceAdapter.ReduceOp(dataLocal, dataLocal, dataLocal, dataCount);
        }
        pipe_barrier(PIPE_V);
        reduceAdapter.AfterReduceOp(outLocal, dataLocal, 1);
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void ReduceCompute(ReduceAdapter<P, OpPredicate> &reduceAdapter,
        LocalTensor<float> &dataLocal, LocalTensor<float> &outLocal, LocalTensor<P>& workLocal, int64_t dataCount) {
        pipe_barrier(PIPE_V);
        reduceAdapter.ReduceOp(dataLocal, outLocal, workLocal, dataCount);
        pipe_barrier(PIPE_V);
    }
};

template <typename T, typename P, typename OpPredicate, int32_t bufferNum=BUFFER_NUM, uint8_t paramsCount=INPUT_PARAMETER_COUNT, typename Tiling=ForeachReduceTilingData>
class  ForeachReduceUnary: public KernelForeadhReduce<T, P, ForeachReduceUnary<T, P, OpPredicate, bufferNum, paramsCount, Tiling>, bufferNum, paramsCount, Tiling> {
public:
    using Base = KernelForeadhReduce<T, P, ForeachReduceUnary<T, P, OpPredicate, bufferNum, paramsCount, Tiling>, bufferNum, paramsCount, Tiling>;

    __aicore__ inline ForeachReduceUnary(OpPredicate &op) : Base(*this), reduceAdapter(op){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const Tiling* tilingData);
    using Base::Process;

private:
    InnerComputer<T, P, OpPredicate> computer;
    ReduceAdapter<P, OpPredicate> reduceAdapter;
private:
    __aicore__ inline void ComputeRound1(uint16_t index, int64_t dataCount, LocalTensor<P> & tempLocal) {
        LocalTensor<T> dataLocal = Base::dataQueue.template DeQue<T>();

        computer.ComputeRound1(
            reduceAdapter,
            dataLocal,
            tempLocal,
            Base::float32Tensor,
            Base::maxCastDataCount,
            dataCount,
            index);

        Base::dataQueue.FreeTensor(dataLocal);
    }

    __aicore__ inline void ComputeRound2(uint16_t dataCount, uint16_t offset) {
        LocalTensor<P> dataLocal = Base::dataQueue.template DeQue<P>();
        LocalTensor<T> outLocal = Base::outQueue.template AllocTensor<T>();

        computer.ComputeRound2(reduceAdapter, dataLocal, outLocal, dataCount);

        event_t eventID1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventID1);
        WaitFlag<HardEvent::V_MTE3>(eventID1);

        DataCopyExtParams copyParams2{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位        
        DataCopyPad(Base::outTensorGM, outLocal, copyParams2);

        event_t eventID2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventID2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventID2);

        Base::dataQueue.FreeTensor(dataLocal);
        Base::outQueue.FreeTensor(outLocal);
    }

    __aicore__ inline void ReduceCompute(LocalTensor<P>& dstLocal, LocalTensor<P>& srcLocal, LocalTensor<P>& workLocal, int32_t count) {
        computer.ReduceCompute(reduceAdapter, dstLocal, srcLocal, workLocal, count);
    }

    __aicore__ inline void BeforeProcess() {
    }

    __aicore__ inline void AfterProcess() {
    }

    __aicore__ inline void CopyInPlusStage1(uint32_t index, int64_t dataCount) {
    }

    __aicore__ inline bool CopyOut(uint32_t index, int64_t dataCount) {
        return false;
    }

    __aicore__ inline void ProcessPlusInLoop(uint32_t index, uint64_t cursorStart) {}

    friend Base;
};

template <typename T, typename P, typename OpPredicate, int32_t bufferNum, uint8_t paramsCount, typename Tiling>
__aicore__ inline void ForeachReduceUnary<T, P, OpPredicate, bufferNum, paramsCount, Tiling>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                const Tiling* tilingData) {                           
    Base::Init(x, y, workspace, tilingData);
}

}  // namespace OpKernel
}  // namespace Common

#endif  // KERNEL_FOREACH_REDUCE_ONE_SCALAR_BINARY_H