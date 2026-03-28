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
 * @file scatter_reduce_spec1.h
 */
#ifndef SCATTER_REDUCE_SPEC1_H 
#define SCATTER_REDUCE_SPEC1_H

#include "kernel_operator.h"
using namespace AscendC;


class ScatterDeduceSpec1 {
   public:
   static constexpr int BLOCK_BYTES_SIZE = 32;
   static constexpr int MAX_VALUE = 10000;
   static constexpr int BUFFER_NUM = 1;
    __aicore__ inline ScatterDeduceSpec1() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR index, GM_ADDR src, GM_ADDR y, int32_t batchSize,
                                int32_t dimSizeX, int32_t dimSizeSrc, int32_t strideSize,
                                int32_t reduction, bool includeSelf, TPipe* pipeIn) {
#ifdef PRINTF
        printf("in spec1 init\n");
#endif
        this->coreNum = GetBlockNum();
        this->coreIndex = GetBlockIdx();
        this->pipe = pipeIn;

        this->batchSize = batchSize;
        this->dimSizeX = dimSizeX;
        this->dimSizeSrc = dimSizeSrc;
        this->strideSize = strideSize;
        this->reduction = reduction;
        this->includeSelf = includeSelf;
        this->totalSizeX = batchSize * dimSizeX * strideSize;
        this->totalSizeSrc = batchSize * dimSizeSrc * strideSize;

        this->BlockSize = BLOCK_BYTES_SIZE / sizeof(DTYPE_Y);
        this->BlockTotalNum = (strideSize + BlockSize - 1) / BlockSize;
        int BlockMod = BlockTotalNum % coreNum;

        this->BlockNum = (BlockTotalNum / coreNum) + (coreIndex < BlockMod ? 1 : 0);
        this->BlockBegin = (BlockTotalNum / coreNum) * coreIndex +
                           (coreIndex < BlockMod ? coreIndex : BlockMod);
        this->totalBlockSizeRow = ((BLOCK_BYTES_SIZE * dimSizeX + 255) / 256) * 256 /
                                  sizeof(DTYPE_Y);

        xGm.SetGlobalBuffer((__gm__ DTYPE_Y*)x, totalSizeX * sizeof(DTYPE_Y));
        indexGm.SetGlobalBuffer((__gm__ DTYPE_INDEX*)index, totalSizeSrc * sizeof(DTYPE_INDEX));
        srcGm.SetGlobalBuffer((__gm__ DTYPE_Y*)src, totalSizeSrc * sizeof(DTYPE_Y));
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y, totalSizeX * sizeof(DTYPE_Y));

        pipe->InitBuffer(InX, BUFFER_NUM, dimSizeX * BLOCK_BYTES_SIZE + 256);
        pipe->InitBuffer(InIndex, BUFFER_NUM, dimSizeX * BLOCK_BYTES_SIZE);
        pipe->InitBuffer(InSrc, BUFFER_NUM, dimSizeX * BLOCK_BYTES_SIZE);
        pipe->InitBuffer(OutY, BUFFER_NUM, dimSizeX * BLOCK_BYTES_SIZE + 256);

        pipe->InitBuffer(VectorIdxBuf, BLOCK_BYTES_SIZE);
        pipe->InitBuffer(CalIdxBuf, BLOCK_BYTES_SIZE);
        pipe->InitBuffer(TmpYBuf, BLOCK_BYTES_SIZE);

        pipe->InitBuffer(CmpBuf, totalBlockSizeRow / sizeof(DTYPE_Y) + 256);

        auto vectorIdxBuf = VectorIdxBuf.Get<DTYPE_INDEX>();

#if USE_VEC_INDEX
        CreateVecIndex(vectorIdxBuf, 0, BlockSize);
        Muls(vectorIdxBuf, vectorIdxBuf, (DTYPE_INDEX)4, BlockSize);
#else
        for (int i = 0; i < BlockSize; i++) {
            vectorIdxBuf.SetValue(i, i * 4);
        }
#endif
#ifdef PRINTF
        printf(
            "batchSize=%d, dimSizeX=%d, dimSizeSrc=%d, strideSize=%d, reduction=%d, "
            "includeSelf=%d\n",
            batchSize, dimSizeX, dimSizeSrc, strideSize, reduction, includeSelf);
        printf(
            "coreNum=%d, coreIndex=%d, BlockSize=%d, BlockTotalNum=%d, BlockNum=%d, "
            "BlockBegin=%d\n",
            coreNum, coreIndex, BlockSize, BlockTotalNum, BlockNum, BlockBegin);
#endif
    }

    __aicore__ inline void Process() {
        int end = (BlockBegin + BlockNum) * BlockSize;
        end = strideSize < end ? strideSize : end;
#ifdef PRINTF
        printf("begin=%d, end=%d\n", BlockBegin * BlockSize, end);
#endif
        for (int i = BlockBegin * BlockSize; i < end; i += BlockSize) {
            int calNum = (end - i) < BlockSize ? (end - i) : BlockSize;
            CopyIn(i, calNum);
            Compute(i, calNum);
            CopyOut(i, calNum);
        }
    }

    __aicore__ inline void CopyIn(int begin, int calNum) {
        auto inXLocal = InX.AllocTensor<DTYPE_Y>();
        auto inIndexLocal = InIndex.AllocTensor<DTYPE_INDEX>();
        auto inSrcLocal = InSrc.AllocTensor<DTYPE_Y>();
        DataCopyExtParams copyParams{
            (uint16_t)dimSizeX, static_cast<uint32_t>(calNum * sizeof(DTYPE_INDEX)),
            static_cast<uint32_t>((strideSize - calNum) * sizeof(DTYPE_INDEX)), 0, 0};
        DataCopyPadExtParams<DTYPE_INDEX> padParams{false, 0, 0, 0};
        DataCopyPad(inIndexLocal, indexGm[begin], copyParams, padParams);

        DataCopyPadExtParams<DTYPE_Y> padParams1{false, 0, 0, 0};
        DataCopyPad(inSrcLocal, srcGm[begin], copyParams, padParams1);
        DataCopyPad(inXLocal, xGm[begin], copyParams, padParams1);
        InX.EnQue(inXLocal);
        InIndex.EnQue(inIndexLocal);
        InSrc.EnQue(inSrcLocal);
    }

    __aicore__ inline void Compute(int begin, int calNum) {
        auto outYLocal = OutY.AllocTensor<DTYPE_Y>();
        Duplicate(outYLocal, (DTYPE_Y)MAX_VALUE, BlockSize * dimSizeX);
        auto vectorIdxBuf = VectorIdxBuf.Get<DTYPE_INDEX>();
        auto calIdxBuf = CalIdxBuf.Get<DTYPE_INDEX>();
        auto calIdxBufUint = calIdxBuf.ReinterpretCast<uint32_t>();
        auto tmpYBuf = TmpYBuf.Get<DTYPE_Y>();
        auto cmpBuf = CmpBuf.Get<uint8_t>();

        auto inIndexLocal = InIndex.DeQue<DTYPE_INDEX>();
        auto inSrcLocal = InSrc.DeQue<DTYPE_Y>();
        auto inXLocal = InX.DeQue<DTYPE_Y>();
#ifdef PRINTF
        printf("inIndexLocal:\n");
        PrintMartixIdx(inIndexLocal, dimSizeX, calNum, 8);

        for (int i = 0; i < 40; i++) {
            printf("%d ", inIndexLocal.GetValue(i));
        }
        printf("\n inXLocal:\n");
        PrintMartixY(inXLocal, dimSizeX, calNum, 8);

        printf("inSrcLocal:\n");
        PrintMartixY(inSrcLocal, dimSizeX, calNum, 8);

        printf("outYLocal:\n");
        PrintMartixY(outYLocal, dimSizeX, calNum, 8);
#endif

        for (int i = 0; i < dimSizeX; i++) {
#ifdef PRINTF
            printf(" \n\n i=%d: calNum=%d \n", i, calNum);
#endif

            Muls(calIdxBuf, inIndexLocal[i * BlockSize], (DTYPE_INDEX)BLOCK_BYTES_SIZE, calNum);
            Add(calIdxBuf, calIdxBuf, vectorIdxBuf, calNum);

            Gather(tmpYBuf, outYLocal, calIdxBufUint, (uint32_t)0, (uint32_t)calNum);

            Min(tmpYBuf, tmpYBuf, inSrcLocal[i * BlockSize], calNum);

            ShiftRight(calIdxBuf, calIdxBuf, 2, calNum);
            int j=0;
            for (; j < calNum-3; j+=4) {
                int tmp1 = calIdxBuf.GetValue(j);
                int tmp2 = calIdxBuf.GetValue(j+1);
                int tmp3 = calIdxBuf.GetValue(j+2);
                int tmp4 = calIdxBuf.GetValue(j+3);
                DTYPE_Y tmpy1 = tmpYBuf.GetValue(j);
                DTYPE_Y tmpy2 = tmpYBuf.GetValue(j+1);
                DTYPE_Y tmpy3 = tmpYBuf.GetValue(j+2);
                DTYPE_Y tmpy4 = tmpYBuf.GetValue(j+3);
                outYLocal.SetValue(tmp1, tmpy1);
                outYLocal.SetValue(tmp2, tmpy2);
                outYLocal.SetValue(tmp3, tmpy3);
                outYLocal.SetValue(tmp4, tmpy4);
            }
            while (j<calNum)
            {
                int tmp = calIdxBuf.GetValue(j);
                outYLocal.SetValue(tmp, tmpYBuf.GetValue(j));
                j++;
            }
        }

        CompareScalar(cmpBuf, outYLocal, (DTYPE_Y)MAX_VALUE, CMPMODE::EQ, totalBlockSizeRow);
        Select(outYLocal, cmpBuf, inXLocal, outYLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
               totalBlockSizeRow);

        OutY.EnQue(outYLocal);

        InIndex.FreeTensor(inIndexLocal);
        InSrc.FreeTensor(inSrcLocal);
        InX.FreeTensor(inXLocal);
    }

    __aicore__ inline void CopyOut(int begin, int calNum) {
        auto outYLocal = OutY.DeQue<DTYPE_Y>();
        DataCopyParams copyParams{(uint16_t)dimSizeX,
                                  static_cast<uint16_t>(calNum * sizeof(DTYPE_Y)), 0,
                                  static_cast<uint16_t>((strideSize - calNum) * sizeof(DTYPE_Y))};
        DataCopyPad(yGm[begin], outYLocal, copyParams);

        OutY.FreeTensor(outYLocal);
    }

    __aicore__ inline void PrintMartixY(LocalTensor<DTYPE_Y>& x, int col, int row, int ld) {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                printf("%f ", x.GetValue(i * ld + j));
            }
            printf("\n");
        }
    }
    __aicore__ inline void PrintMartixIdx(LocalTensor<DTYPE_INDEX>& x, int col, int row, int ld) {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                printf("%d ", x.GetValue(i * ld + j));
            }
            printf("\n");
        }
    }

   private:
    TPipe* pipe;
    GlobalTensor<DTYPE_Y> xGm;
    GlobalTensor<DTYPE_INDEX> indexGm;
    GlobalTensor<DTYPE_Y> srcGm;
    GlobalTensor<DTYPE_Y> yGm;

    bool includeSelf;
    TBuf<QuePosition::VECCALC> TmpYBuf, VectorIdxBuf, CalIdxBuf, CmpBuf;
    TQue<QuePosition::VECIN, BUFFER_NUM> InX, InIndex, InSrc;
    TQue<QuePosition::VECOUT, BUFFER_NUM> OutY;

    int32_t batchSize, dimSizeX, dimSizeSrc, strideSize, reduction, totalSizeX, totalSizeSrc;
    uint32_t coreIndex, coreNum;
    uint32_t BlockSize, BlockTotalNum, BlockNum, BlockBegin, totalBlockSizeRow;
};

#endif // SCATTER_REDUCE_SPEC1_H