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
 * @file radius.cpp
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename TYPE_X>
class KernelRadius
{
    using T = TYPE_X;
public:
    __aicore__ inline KernelRadius() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR ptr_x, GM_ADDR ptr_y, GM_ADDR out,
                                float r, uint32_t ignoreSameIndex, uint32_t maxNumNeighbors,
                                uint32_t xSize, uint32_t ySize, uint32_t itemLength, uint32_t ptrXLen, uint32_t ptrYLen)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->r = r * r;
        this->ignoreSameIndex = ignoreSameIndex;
        this->maxNumNeighbors = maxNumNeighbors;
        this->xSize = xSize;
        this->ySize = ySize;
        this->itemLength = itemLength;
        this->maxSize = maxNumNeighbors * ySize;
        this->totalCount = 0;
        this->ptrXLen = ptrXLen;
        this->ptrYLen = ptrYLen;
        xGm.SetGlobalBuffer((__gm__ TYPE_X *)x, xSize * itemLength);
        yGm.SetGlobalBuffer((__gm__ TYPE_X *)y, ySize * itemLength);
        outGm.SetGlobalBuffer((__gm__ TYPE_X *)out, maxSize * 2);
        if(ptrXLen!=0){
            ptrXGm.SetGlobalBuffer((__gm__ int32_t *)ptr_x, ptrXLen);
        }
        if(ptrYLen!=0){
            ptrYGm.SetGlobalBuffer((__gm__ int32_t *)ptr_y, ptrYLen);
        }
        pipe.InitBuffer(itemBuf, AlignUp(this->itemLength) * sizeof(TYPE_X));
        pipe.InitBuffer(xItemBuf, AlignUp(this->itemLength) * sizeof(TYPE_X));
        pipe.InitBuffer(xItemFpBuf, AlignUp(this->itemLength) * sizeof(float));
        pipe.InitBuffer(itemFpBuf, AlignUp(this->itemLength) * sizeof(float));
        pipe.InitBuffer(resBuf, AlignUp(this->itemLength) * sizeof(int8_t));
    }
    __aicore__ inline int AlignUp(int x){
        return (x + 31) / 32 * 32;
    }

    __aicore__ inline void Process()
    {
        // Handle the two main cases: with or without batch pointers
        if (ptrYLen == 0) {
            // Case 1: No batch pointers. Process all x vs all y.
            for (int i = 0; i < ySize; i++) {
                ProcessNeighbors(i, 0, xSize);
            }
        } else {
            // Case 2: Batch pointers are provided.
            int32_t yEnd = ptrYGm.GetValue(0);
            int32_t xEnd = ptrXGm.GetValue(0);
            for (int q = 0; q < ptrYLen - 1; q++) {
                int32_t yStart = yEnd;
                int32_t xStart = xEnd;
                yEnd = ptrYGm.GetValue(q + 1);
                xEnd = ptrXGm.GetValue(q + 1);
                if (yStart == yEnd || xStart == xEnd) {
                    continue;
                }
                for (int i = yStart; i < yEnd; i++) {
                    ProcessNeighbors(i, xStart, xEnd);
                }
            }
        }

        // Final copy of indices.
        if (totalCount < maxSize) {
            for (int i = 0; i < totalCount; i++) {
                outGm.SetValue(totalCount + i, (TYPE_X)outGm.GetValue(maxSize + i));
            }
        }
    }

private:
    // Helper function to check if two points are within the radius.
    // This function encapsulates the type-specific calculation logic and removes code duplication.
    __aicore__ inline bool IsInRadius(LocalTensor<TYPE_X>& xItem, LocalTensor<TYPE_X>& yItem)
    {
        LocalTensor<int8_t> resLocal = resBuf.Get<int8_t>();
        uint64_t mask[2] = { 1, 0 };
        UnaryRepeatParams repeatParams = { 1, 1, 8, 8 };

        if constexpr (std::is_same_v<TYPE_X, float>) {
            Sub(xItem, xItem, yItem, AlignUp(itemLength));
            Mul(xItem, xItem, xItem, AlignUp(itemLength));
            ReduceSum<float>(xItem, xItem, xItem, itemLength);
            CompareScalar(resLocal, xItem, (float)r, AscendC::CMPMODE::LE, mask, 1, repeatParams);
        } else { // Handles int32_t and half types
            LocalTensor<float> xItemLocalFp = xItemFpBuf.Get<float>();
            LocalTensor<float> itemLocalFp = itemFpBuf.Get<float>();
            
            // Use different rounding modes for int32_t and half
            constexpr auto mode = std::is_same_v<TYPE_X, int32_t> ? RoundMode::CAST_RINT : RoundMode::CAST_NONE;
            Cast(xItemLocalFp, xItem, mode, AlignUp(itemLength));
            Cast(itemLocalFp, yItem, mode, AlignUp(itemLength));

            Sub(xItemLocalFp, xItemLocalFp, itemLocalFp, AlignUp(itemLength));
            Mul(xItemLocalFp, xItemLocalFp, xItemLocalFp, AlignUp(itemLength));
            ReduceSum<float>(xItemLocalFp, xItemLocalFp, xItemLocalFp, itemLength);
            CompareScalar(resLocal, xItemLocalFp, (float)r, AscendC::CMPMODE::LE, mask, 1, repeatParams);
        }

        int8_t value = resLocal.GetValue(0);
        return (value & 0b00000001) != 0;
    }

    // Helper function to process all neighbors for a given point y[i].
    // This reduces the nesting depth in the main Process function.
    __aicore__ inline void ProcessNeighbors(int32_t i, int32_t xStart, int32_t xEnd)
    {
        LocalTensor<TYPE_X> itemLocal = itemBuf.Get<TYPE_X>();
        // Load the point y[i] into local memory
        for (int k = 0; k < itemLength; k++) {
            itemLocal.SetValue(k, (TYPE_X)yGm.GetValue(i * itemLength + k));
        }

        int32_t count = 0;
        for (int j = xStart; j < xEnd; j++) {
            if (j == i && ignoreSameIndex) {
                continue;
            }

            LocalTensor<TYPE_X> xItemLocal = xItemBuf.Get<TYPE_X>();
            // Load the point x[j] into local memory
            for (int k = 0; k < itemLength; k++) {
                xItemLocal.SetValue(k, (TYPE_X)xGm.GetValue(j * itemLength + k));
            }

            if (IsInRadius(xItemLocal, itemLocal)) {
                outGm.SetValue(totalCount, (TYPE_X)j);
                outGm.SetValue(totalCount + maxSize, (TYPE_X)i);
                count++;
                totalCount++;
                if (count == maxNumNeighbors) {
                    break; // Found max neighbors for point i
                }
            }
        }
    }

    TPipe pipe;
    GlobalTensor<TYPE_X> xGm, yGm, outGm;
    GlobalTensor<int32_t> ptrXGm, ptrYGm;
    TBuf<QuePosition::VECCALC> itemBuf, xItemBuf, itemFpBuf, xItemFpBuf, resBuf;
    float r;
    uint32_t ignoreSameIndex, maxNumNeighbors;
    uint32_t ySize, itemLength, totalCount, xSize, maxSize, ptrXLen, ptrYLen;
};

extern "C" __global__ __aicore__ void radius(GM_ADDR x, GM_ADDR y, GM_ADDR ptr_x, GM_ADDR ptr_y, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelRadius<DTYPE_X> op;
    op.Init(x, y, ptr_x, ptr_y, out,
        tiling_data.r, tiling_data.ignore_same_index, tiling_data.max_num_neighbors,
        tiling_data.xSize, tiling_data.ySize, tiling_data.itemLength, tiling_data.ptrXLen, tiling_data.ptrYLen);  
    op.Process();
}