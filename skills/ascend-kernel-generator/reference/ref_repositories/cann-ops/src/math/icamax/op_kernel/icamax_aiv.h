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
 * @file icamax_aiv.h
 */
#ifndef ICAMAX_H
#define ICAMAX_H

#include <type_traits>
#include "kernel_operator.h"

namespace Icamax {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BYTE_BLOCK = 32;
constexpr int32_t BYTES_PER_REPEAT = 256;
constexpr int32_t MAX_REPEATS = 255;
constexpr int32_t BLOCKS_PER_REPEAT = 8;
constexpr int32_t MAX_CAST_COUNT = 512;
constexpr uint32_t BYTE_LEN_4 = 4;
constexpr uint32_t MAX_VECTOT = 40;
constexpr uint64_t REDUCE_MAX_RST_MASK = 72340172838076673;  // 2^0+2^8+2^16+2^24+2^32+2^40+2^48+2^56

// 超过32超过一个repeat上限，reduceMaz进入第二个repeat，全置零后可复用该mask,i = 0 到 31,REDUCE_MAX_CORES_RST_MASK += (uint64_t)1<<(i*2);
constexpr uint64_t REDUCE_MAX_CORES_RST_MASK = 6148914691236517205;  // (i*2)^0+(i*2)^2+(i*2)^4+..+(i*2)^31

constexpr int32_t GM_RESULT_LEN = 2;

constexpr uint32_t MAX_NUM_F32_ELE_EACH_CORE = 23040;

// 如果输入数据很大，UB需多次处理，并且轮次很多，不能等所有轮次都汇总完，这样中间结果太占内存，需要及时对部分中间结果取reduceMax
// 中间结果按2k规划，即2*1024/32=64次，即单核超过63次,就要取一次reduceMax，将空间占用降为一个block
// 64-63是因为有一个是历史压缩结果
constexpr int32_t DEAL_TIMES_EACH_CORE_REDUCE = 63;
constexpr int32_t DEAL_TIMES_EACH_CORE_REDUCE_REP = 8;  // (63+1)/8=8个repeat

template <typename T>
class Icamax {
public:
    __aicore__ inline Icamax<T>(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR usrWorkspace, GM_ADDR tiling);
    __aicore__ inline void Process();

private:
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilA2B(T1 a, T2 b)
    {
        if (b == 0) {
            return a;
        }
        return (a + b - 1) / b;
    };
    __aicore__ inline bool ParseTilingData(GM_ADDR tiling);
    __aicore__ inline void CopyIn(uint32_t calCount, uint32_t offset);
    __aicore__ inline void CopyInAttacheTail(uint32_t calCount, uint32_t tailCount, uint32_t offset);
    __aicore__ inline void SingleProcess(const int32_t count, uint32_t k);
    __aicore__ inline void GetReduceMaxCount(uint32_t k);
    __aicore__ inline void reduceMaxTmpResult(uint32_t k);
    __aicore__ inline void getCoreTmpReduResult();
    __aicore__ inline void CopyTmpRstToWkGM();
    __aicore__ inline void reduceMaxCoresResult();
    __aicore__ inline void copyInCoresTmpRst();
    __aicore__ inline void ReduceMaxCoresTmpRst();
    __aicore__ inline void copyOutRst();

    __aicore__ inline void getCoreReduResult();
    __aicore__ inline void CopyTmpRstOut();

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inDataQueue;
    TBuf<QuePosition::VECCALC> reduceMaxRsts;
    TBuf<QuePosition::VECCALC> reduceMaxWrk;
    TBuf<QuePosition::VECCALC> reduceMaxDst;
    TBuf<QuePosition::VECCALC> rMaxRsts;

    // outDataQueue这个应该1也够，因为单核内所有数据都压缩成一个结果
    TQue<QuePosition::VECOUT, BUFFER_NUM> outDataQueue;

    TQue<QuePosition::VECIN, 1> coresRstInDataQueue;
    TBuf<QuePosition::VECCALC> coresRstReduceMaxWrk;
    TQue<QuePosition::VECOUT, 1> coresRstOutDataQueue;

    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> tmpRstWkGM;
    GlobalTensor<int32_t> outTensorsGM;

    LocalTensor<T> rMaxALLRstsTenor;
    LocalTensor<T> rMaxTmpRstsTenor;
    LocalTensor<T> rMaxTmpWrkTenor;
    LocalTensor<T> rMaxDstTenor;

    LocalTensor<T> coresRstReduceTmp;
    LocalTensor<int32_t> rMaxRstsTenor;

    uint32_t blockIdx = 0;
    uint32_t elementsPerBlock = BYTE_BLOCK / BYTE_LEN_4;
    uint32_t elementsPerRepeat = BYTES_PER_REPEAT / BYTE_LEN_4;

    uint32_t incx;
    uint32_t needVecCoreNum;
    uint32_t dytpeFlag;
    uint32_t startOffset;
    uint32_t eleTotalEachCore;

    uint32_t dealTimesEachCore;
    uint32_t dealLenEachTime;
    uint32_t reduceMaxRstsLen;
    uint32_t dealLenUpBlockEachTime;
    uint32_t coresRstReduceMaxLen;
    uint32_t rstLenAllCoreBytes;
    uint32_t tailCount;
    uint32_t tailEle;

    uint32_t maxRepeatLen;
    uint32_t totalRptCntNor;
    uint32_t totalRptCntNorRemainder;  // should calc
    uint32_t rptBatchCntNor;           // limit by L0 API, should calc
    uint32_t rptBatchCntNorRemainder;  // should calc
    uint32_t rmdRptLenNor;
    bool isUpdateRptParas = false;

    uint32_t startElement;
    uint32_t dealEleEachTime;
};

template <typename T>
__aicore__ inline void Icamax<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR usrWorkspace, GM_ADDR tiling)
{
    this->blockIdx = GetBlockIdx();

    if (false == ParseTilingData(tiling)) {
        return;
    }

    inTensorsGM.SetGlobalBuffer((__gm__ T *)x + startOffset, eleTotalEachCore);
    tmpRstWkGM.SetGlobalBuffer((__gm__ T *)usrWorkspace + blockIdx * GM_RESULT_LEN, GM_RESULT_LEN);
    outTensorsGM.SetGlobalBuffer((__gm__ int32_t *)y, GM_RESULT_LEN);

    this->pipe.InitBuffer(this->inDataQueue, BUFFER_NUM, this->dealLenUpBlockEachTime * BYTE_LEN_4);
    this->pipe.InitBuffer(this->reduceMaxRsts,
                          this->reduceMaxRstsLen * BYTE_LEN_4);  // 保存多次reduceMax的结果，用于汇总
    this->pipe.InitBuffer(this->reduceMaxWrk, this->reduceMaxRstsLen * BYTE_LEN_4);
    this->pipe.InitBuffer(this->reduceMaxDst, this->elementsPerBlock * BYTE_LEN_4);
    this->pipe.InitBuffer(this->rMaxRsts, this->elementsPerBlock * BYTE_LEN_4);
    this->pipe.InitBuffer(this->outDataQueue, BUFFER_NUM, this->elementsPerBlock * BYTE_LEN_4);

    // 核间结果在GM（每个核都会给出两个值 value、index）上汇总后，再次处理
    // 如果needVecCoreNum大于32个核，则核间结果value、index长度超过64，超过一次repeat，由于needVecCoreNum最大为40（80），所以直接预存两个repeat128，多余空间置零
    this->coresRstReduceMaxLen = this->elementsPerRepeat * 2;
    this->pipe.InitBuffer(this->coresRstInDataQueue, 1, this->coresRstReduceMaxLen * BYTE_LEN_4);
    this->pipe.InitBuffer(this->coresRstReduceMaxWrk, this->coresRstReduceMaxLen * BYTE_LEN_4);
    this->pipe.InitBuffer(this->coresRstOutDataQueue, 1, this->elementsPerBlock * BYTE_LEN_4);

    // 第一个elementsPerBlock汇总一次大轮结果，后续内存用于存每一个小轮循环
    this->rMaxALLRstsTenor = reduceMaxRsts.Get<T>();
    this->rMaxTmpRstsTenor = this->rMaxALLRstsTenor[this->elementsPerBlock];
    this->rMaxTmpWrkTenor = reduceMaxWrk.Get<T>();
    this->rMaxDstTenor = reduceMaxDst.Get<T>();
    this->rMaxRstsTenor = rMaxRsts.Get<int32_t>();
    this->coresRstReduceTmp = coresRstReduceMaxWrk.Get<T>();
}

template <typename T>
__aicore__ inline bool Icamax<T>::ParseTilingData(GM_ADDR tiling)
{
    auto tilingBuf = reinterpret_cast<__gm__ uint32_t *>(tiling);
    this->incx = (*(__gm__ uint32_t *)(tilingBuf + 0));
    this->needVecCoreNum = (*(__gm__ uint32_t *)(tilingBuf + 1));

    // mix模式下，每个AICore会启动两个AIVcore,。tiling中根据AIVcoreNum推算的BlockDim可能会多一个
    if (this->blockIdx + 1 > this->needVecCoreNum) {
        return false;
    }

    this->dytpeFlag = (*(__gm__ uint32_t *)(tilingBuf + 2));
    this->rstLenAllCoreBytes = (*(__gm__ uint32_t *)(tilingBuf + 3));
    this->tailCount = (*(__gm__ uint32_t *)(tilingBuf + 4));
    this->maxRepeatLen = (*(__gm__ uint32_t *)(tilingBuf + 5));

    this->startOffset = (*(__gm__ uint32_t *)(tilingBuf + 6 + this->blockIdx));
    this->eleTotalEachCore = (*(__gm__ uint32_t *)(tilingBuf + 6 + 1 * MAX_VECTOT + this->blockIdx));
    this->dealTimesEachCore = (*(__gm__ uint32_t *)(tilingBuf + 6 + 2 * MAX_VECTOT + this->blockIdx));
    this->dealLenEachTime = (*(__gm__ uint32_t *)(tilingBuf + 6 + 3 * MAX_VECTOT + this->blockIdx));
    this->reduceMaxRstsLen = (*(__gm__ uint32_t *)(tilingBuf + 6 + 4 * MAX_VECTOT + this->blockIdx));
    this->dealLenUpBlockEachTime = (*(__gm__ uint32_t *)(tilingBuf + 6 + 5 * MAX_VECTOT + this->blockIdx));

    this->totalRptCntNor = (*(__gm__ uint32_t *)(tilingBuf + 6 + 6 * MAX_VECTOT + this->blockIdx));
    this->totalRptCntNorRemainder = (*(__gm__ uint32_t *)(tilingBuf + 6 + 7 * MAX_VECTOT + this->blockIdx));
    this->rptBatchCntNor = (*(__gm__ uint32_t *)(tilingBuf + 6 + 8 * MAX_VECTOT + this->blockIdx));
    this->rptBatchCntNorRemainder = (*(__gm__ uint32_t *)(tilingBuf + 6 + 9 * MAX_VECTOT + this->blockIdx));
    this->rmdRptLenNor = (*(__gm__ uint32_t *)(tilingBuf + 6 + 10 * MAX_VECTOT + this->blockIdx));

    this->startElement = this->startOffset;
    this->dealEleEachTime = this->dealLenEachTime;
    this->tailEle = this->tailCount;
    if (this->dytpeFlag == 1) {  // 如果是复数
        this->startElement = this->startOffset / 2;
        this->dealEleEachTime = this->dealEleEachTime / 2;
        this->tailEle = this->tailCount / 2;
    }

    return true;
}

template <typename T>
__aicore__ inline void Icamax<T>::Process()
{
    // 硬同步必须在mix模式下，在该模式下，AIVector会成对出现，blockIdx实际比needVecCoreNum多，直接return空转
    if (this->blockIdx + 1 > this->needVecCoreNum) {
        SyncAll();
        return;
    }
    Duplicate<T>(this->rMaxALLRstsTenor, 0.0, this->reduceMaxRstsLen);

    uint32_t calCount = 0;
    uint32_t copyedlCount = 0;
    int32_t leftCount = 0;
    // 如果dealBigTimestail不为0，说明最后一个大轮不满DEAL_TIMES_EACH_CORE_REDUCE
    int32_t dealBigTimes = this->dealTimesEachCore / DEAL_TIMES_EACH_CORE_REDUCE;
    int32_t dealBigTimesTail = this->dealTimesEachCore % DEAL_TIMES_EACH_CORE_REDUCE;

    // 小轮
    for (uint32_t k = 0; k < this->dealTimesEachCore; k++) {
        // 更新本次要处理的元素数量
        calCount = this->dealLenEachTime;
        this->isUpdateRptParas = false;
        leftCount = this->eleTotalEachCore - copyedlCount;

        if (leftCount < this->dealLenEachTime) {
            calCount = leftCount;
            isUpdateRptParas = true;
        }

        // 尾块全给第一个核了，长度不足一个repeat
        if (this->blockIdx == 0 && this->tailCount > 0 && k == 0) {
            CopyInAttacheTail(calCount, this->tailCount, copyedlCount);
            calCount += this->tailCount;
            isUpdateRptParas = true;
        } else {
            CopyIn(calCount, copyedlCount);
        }

        SingleProcess(calCount, k);
        copyedlCount += calCount;

        // 大轮：即单核超过DEAL_TIMES_EACH_CORE_REDUCE次,就要取一次reduceMax，将空间 rMaxTmpRstsTenor 占用降为一个block
        uint32_t dealedTimes = k + 1;
        if ((dealedTimes % DEAL_TIMES_EACH_CORE_REDUCE == 0 && k > 0) || (k == this->dealTimesEachCore - 1)) {
            reduceMaxTmpResult(k);
            // 如果下一次是进入最后一个大轮，且不满DEAL_TIMES_EACH_CORE_REDUCE,需要清理rMaxTmpRstsTenor；但是大轮中最后一次也会走进来，不需要了
            if ((0 != dealBigTimesTail) && (dealedTimes / DEAL_TIMES_EACH_CORE_REDUCE == dealBigTimes) &&
                dealedTimes != this->dealTimesEachCore) {
                Duplicate<T>(this->rMaxTmpRstsTenor, 0.0, this->reduceMaxRstsLen - this->elementsPerBlock);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }

    if (this->needVecCoreNum == 1 && this->blockIdx == 0) {
        // rMaxALLRstsTenor是单核中间结果，结构为[value，index],如果只有一个核，不需要核间汇总,直接把这个index当成结果考出去
        SyncAll();
        getCoreReduResult();
        CopyTmpRstOut();
    } else {
        getCoreTmpReduResult();
        CopyTmpRstToWkGM();
        SyncAll();
        if (this->blockIdx == 0) {
            reduceMaxCoresResult();
        }
    }
}

template <typename T>
__aicore__ inline void Icamax<T>::CopyIn(uint32_t calCount, uint32_t offset)
{
    LocalTensor<T> inLocalTensor = inDataQueue.AllocTensor<T>();
    DataCopy(inLocalTensor, inTensorsGM[offset], calCount);
    inDataQueue.EnQue(inLocalTensor);
}

template <typename T>
__aicore__ inline void Icamax<T>::CopyInAttacheTail(uint32_t calCount, uint32_t tailCount, uint32_t offset)
{
    LocalTensor<T> inLocalTensor = inDataQueue.AllocTensor<T>();

    if (calCount > 0) {
        DataCopy(inLocalTensor, inTensorsGM[offset], calCount);
    }

    // blockLen长度有限制，
    uint16_t blockCout = (uint16_t)1;
    uint16_t blockLen = tailCount * BYTE_LEN_4;
    DataCopyParams copyParams{blockCout, blockLen, 0, 0};
    DataCopyPadParams padParams{true, 0, 0, 0};  // dummy自动补齐32B，paddingValue=0
    DataCopyPad(inLocalTensor[calCount], inTensorsGM[offset + calCount], copyParams, padParams);

    inDataQueue.EnQue(inLocalTensor);
}

template <typename T>
__aicore__ inline void Icamax<T>::SingleProcess(const int32_t count, uint32_t k)
{
    LocalTensor<T> srcLocal = inDataQueue.DeQue<T>();

    // 如果本次是第64次（k=63）,k-1结果已经在376行处理过了，不需要再处理
    if (k % DEAL_TIMES_EACH_CORE_REDUCE != 0 && k > 0) {  // 上一次
        GetReduceMaxCount(k - 1);
    }

    uint32_t totalRepeatCnt = this->totalRptCntNor;
    uint32_t totalRepeatCntRemainder = this->totalRptCntNorRemainder;  // should calc
    uint32_t repeatBatchCnt = this->rptBatchCntNor;                    // limit by L0 API, should calc
    uint32_t repeatBatchCntRemainder = this->rptBatchCntNorRemainder;  // should calc
    uint32_t maxRepeatLen = this->maxRepeatLen;
    uint32_t rmdRepeatLen = this->rmdRptLenNor;
    if (this->isUpdateRptParas == true) {
        totalRepeatCnt = count / elementsPerRepeat;
        totalRepeatCntRemainder = count % elementsPerRepeat;     // should calc
        repeatBatchCnt = totalRepeatCnt / MAX_REPEATS;           // limit by L0 API, should calc
        repeatBatchCntRemainder = totalRepeatCnt % MAX_REPEATS;  // should calc
        rmdRepeatLen = repeatBatchCntRemainder * elementsPerRepeat;
    }

    uint32_t offset = 0;

    for (uint32_t i = 0; i < repeatBatchCnt; i++) {
        Abs(srcLocal[offset], srcLocal[offset], elementsPerRepeat, MAX_REPEATS, {1, 1, 8, 8});
        offset += maxRepeatLen;
    }

    if (repeatBatchCntRemainder > 0) {
        Abs(srcLocal[offset], srcLocal[offset], elementsPerRepeat, repeatBatchCntRemainder, {1, 1, 8, 8});
        offset += rmdRepeatLen;
    }

    if (totalRepeatCntRemainder > 0) {
        Abs(srcLocal[offset], srcLocal[offset], totalRepeatCntRemainder, 1, {1, 1, 8, 8});
    }

    AscendC::PipeBarrier<PIPE_V>();

    int32_t ReduceMaxCount = count;
    // 如果是复数
    if (this->dytpeFlag == 1) {
        ReduceMaxCount = count / 2;
        offset = 0;
        for (uint32_t i = 0; i < repeatBatchCnt; i++) {
            PairReduceSum(srcLocal[offset / 2], srcLocal[offset], MAX_REPEATS, elementsPerRepeat, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            offset += maxRepeatLen;
        }

        if (repeatBatchCntRemainder > 0) {
            PairReduceSum(srcLocal[offset / 2], srcLocal[offset], repeatBatchCntRemainder, elementsPerRepeat, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            offset += rmdRepeatLen;
        }

        if (totalRepeatCntRemainder > 0) {
            PairReduceSum(srcLocal[offset / 2], srcLocal[offset], 1, totalRepeatCntRemainder, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    AscendC::SetMaskCount();
    AscendC::SetVectorMask<T, MaskMode::COUNTER>(0, ReduceMaxCount);
    WholeReduceMax<T, false>(srcLocal, srcLocal, AscendC::MASK_PLACEHOLDER, 1, 1, 1, 8);
    AscendC::SetMaskNorm();
    AscendC::ResetMask();

    // 如果是第63次（k=62）或者是最后一次,本循环内马上就要进行压缩，此时必须把本次结果取出来参与压缩，不能等下次再处理；
    if (((k + 1) % DEAL_TIMES_EACH_CORE_REDUCE == 0 && k > 0) || k == this->dealTimesEachCore - 1) {
        GetReduceMaxCount(k);
    }

    inDataQueue.FreeTensor(srcLocal);
}

template <typename T>
__aicore__ inline void Icamax<T>::GetReduceMaxCount(uint32_t k)
{
    uint32_t tmpK = k % DEAL_TIMES_EACH_CORE_REDUCE * this->elementsPerBlock;

    // S等V，将这动作提前，放在SingleProcess最后会影响double buffer的MTE2指令的提前发射
    event_t eventID2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventID2);
    WaitFlag<HardEvent::V_S>(eventID2);

    float val = 0;                      // 最值
    float idx = 0;                      // 最值的索引值，
    GetReduceMaxMinCount<T>(val, idx);  // 获取上次WholeReudceMax的结果

    this->rMaxTmpRstsTenor.SetValue(tmpK, val);
    this->rMaxTmpRstsTenor.SetValue(tmpK + 1, idx);
}

template <typename T>
__aicore__ inline void Icamax<T>::reduceMaxTmpResult(uint32_t k)
{
    AscendC::PipeBarrier<PIPE_V>();

    uint64_t mask[2] = {REDUCE_MAX_RST_MASK, 0};

    ReduceMax(this->rMaxDstTenor, this->rMaxALLRstsTenor, this->rMaxTmpWrkTenor, mask, DEAL_TIMES_EACH_CORE_REDUCE_REP,
              8, true);

    event_t eventID1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventID1);
    WaitFlag<HardEvent::V_S>(eventID1);

    float tmpK = this->rMaxDstTenor.GetValue(1);                 // 第二次reduceMax的结果，
    int32_t maxIdxInRst = *reinterpret_cast<uint32_t *>(&tmpK);  //

    float tmpK2 = this->rMaxALLRstsTenor.GetValue(maxIdxInRst + 1);
    int32_t maxIdxInOriVec = *reinterpret_cast<uint32_t *>(&tmpK2);

    // rMaxALLRstsTenord的结构为  第一个block为历史压缩结果，第二block开始为本轮第一个结果
    if (maxIdxInRst != 0) {  // 如果不是历史压缩结果最大;
        int32_t preOffset = (k / DEAL_TIMES_EACH_CORE_REDUCE) * DEAL_TIMES_EACH_CORE_REDUCE;
        int32_t maxValueIndex = this->startElement +
                                (preOffset + maxIdxInRst / this->elementsPerBlock - 1) * this->dealEleEachTime +
                                maxIdxInOriVec;
        // 如果且有尾块，k=0处理的长度是dealEleEachTime+tailCount,计算偏移时，除了k=0（maxIdxInRst > 8）
        // 或者k > DEAL_TIMES_EACH_CORE_REDUCE都要加上tailCount
        if (this->tailEle != 0 && this->blockIdx == 0 &&
            (maxIdxInRst > this->elementsPerBlock || k > DEAL_TIMES_EACH_CORE_REDUCE)) {
            maxValueIndex += this->tailEle;
        }
        // 写回头部
        this->rMaxALLRstsTenor.SetValue(0, this->rMaxDstTenor.GetValue(0));
        this->rMaxALLRstsTenor.SetValue(1, *reinterpret_cast<float *>(&maxValueIndex));

        // V等S，V包括getCoreTmpReduResult中的Copy或者SingleProcess中的Abs(更准确的是下一次的reduceMaxTmpResult中的ReduceMax)
        event_t eventID2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventID2);
        WaitFlag<HardEvent::S_V>(eventID2);
    }
}

template <typename T>
__aicore__ inline void Icamax<T>::getCoreTmpReduResult()
{
    LocalTensor<T> outLocalTensor = outDataQueue.AllocTensor<T>();

    // 前8个是压缩的最终结果
    Copy(outLocalTensor, rMaxALLRstsTenor, 8, 1, {1, 1, 8, 8});

    outDataQueue.EnQue(outLocalTensor);
}

template <typename T>
__aicore__ inline void Icamax<T>::CopyTmpRstToWkGM()
{
    LocalTensor<T> outLocalTensor = outDataQueue.DeQue<T>();
    uint16_t blockCout = (uint16_t)1;
    uint16_t blockLen = GM_RESULT_LEN * BYTE_LEN_4;
    DataCopyParams copyParams = {blockCout, blockLen, 0, 0};
    DataCopyPad(tmpRstWkGM, outLocalTensor, copyParams);

    outDataQueue.FreeTensor(outLocalTensor);
}

template <typename T>
__aicore__ inline void Icamax<T>::reduceMaxCoresResult()
{
    copyInCoresTmpRst();
    ReduceMaxCoresTmpRst();
    copyOutRst();
}

template <typename T>
__aicore__ inline void Icamax<T>::copyInCoresTmpRst()
{
    LocalTensor<T> inLocalTensor = coresRstInDataQueue.AllocTensor<T>();

    Duplicate<T>(inLocalTensor, 0.0, this->coresRstReduceMaxLen);  // 多申请了一部分内存置零

    event_t eventID1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventID1);
    WaitFlag<HardEvent::V_MTE2>(eventID1);

    uint16_t blockLen = this->rstLenAllCoreBytes;
    DataCopyParams copyParams{1, blockLen, 0, 0};
    DataCopyPadParams padParams{true, 0, 0, 0};  // dummy自动补齐32B，paddingValue=0
    DataCopyPad(inLocalTensor, tmpRstWkGM, copyParams, padParams);

    coresRstInDataQueue.EnQue(inLocalTensor);
}

template <typename T>
__aicore__ inline void Icamax<T>::ReduceMaxCoresTmpRst()
{
    LocalTensor<T> srcLocal = coresRstInDataQueue.DeQue<T>();
    LocalTensor<int32_t> outLocalTensor = coresRstOutDataQueue.AllocTensor<int32_t>();

    uint64_t mask[2] = {REDUCE_MAX_CORES_RST_MASK, 0};

    // repeatTime = 2，40个vectorCore 最多80个元素
    ReduceMax(this->coresRstReduceTmp, srcLocal, this->coresRstReduceTmp, mask, 2, 8, true);

    event_t eventID3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventID3);
    WaitFlag<HardEvent::V_S>(eventID3);

    float tmpK = this->coresRstReduceTmp.GetValue(1);  // 第二次reduceMax的结果，
    int32_t maxIdxInRst = *reinterpret_cast<uint32_t *>(&tmpK);
    float tmpK2 = srcLocal.GetValue(maxIdxInRst + 1);  // 去第一次汇总结果中找到真正的index
    int32_t maxValueIndex = *reinterpret_cast<uint32_t *>(&tmpK2) + 1;  // 再加1（culblas下标从1开始）

    outLocalTensor.SetValue(0, maxValueIndex);

    event_t eventIDSToMTE4 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIDSToMTE4);
    WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE4);

    coresRstOutDataQueue.EnQue(outLocalTensor);
    coresRstInDataQueue.FreeTensor(srcLocal);
}

template <typename T>
__aicore__ inline void Icamax<T>::copyOutRst()
{
    LocalTensor<int32_t> outLocalTensor = coresRstOutDataQueue.DeQue<int32_t>();

    uint16_t blockCout = (uint16_t)1;
    uint16_t blockLen = BYTE_LEN_4;
    DataCopyParams copyParams = {blockCout, blockLen, 0, 0};
    DataCopyPad(outTensorsGM, outLocalTensor, copyParams);
    coresRstOutDataQueue.FreeTensor(outLocalTensor);
}

template <typename T>
__aicore__ inline void Icamax<T>::getCoreReduResult()
{
    LocalTensor<int32_t> outLocalTensor = coresRstOutDataQueue.AllocTensor<int32_t>();

    float tmpK = rMaxALLRstsTenor.GetValue(1);
    int32_t maxIdxInRst = *reinterpret_cast<uint32_t *>(&tmpK);  //
    rMaxRstsTenor.SetValue(0, maxIdxInRst + 1);                  // 再加1（culblas下标从1开始）

    event_t eventID2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventID2);
    WaitFlag<HardEvent::S_V>(eventID2);

    Copy(outLocalTensor, rMaxRstsTenor, 1, 1, {1, 1, 8, 8});

    coresRstOutDataQueue.EnQue(outLocalTensor);
}

template <typename T>
__aicore__ inline void Icamax<T>::CopyTmpRstOut()
{
    LocalTensor<int32_t> outLocalTensor = coresRstOutDataQueue.DeQue<int32_t>();

    uint16_t blockCout = (uint16_t)1;
    uint16_t blockLen = (uint16_t)1 * BYTE_LEN_4;
    DataCopyParams copyParams = {blockCout, blockLen, 0, 0};
    // coresRstOutDataQueue单核中间结果，结构为[value，index],如果只有一个核，直接把这个index当成结果考出去
    DataCopyPad(outTensorsGM, outLocalTensor, copyParams);

    coresRstOutDataQueue.FreeTensor(outLocalTensor);
}

}  // namespace ICAMAX

#endif  // ICAMAX_H