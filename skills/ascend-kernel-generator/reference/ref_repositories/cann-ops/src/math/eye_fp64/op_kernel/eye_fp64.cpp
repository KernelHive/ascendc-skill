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
 * @file eye_fp64.cpp
 */
#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
using namespace AscendC;
__aicore__ inline uint64_t rotate_left_64(uint64_t value, unsigned int shift) {
    return (value << (shift % 64)) | (value >> (64 - (shift % 64)));
}
__aicore__ inline uint64_t keep_low_y_bits(uint64_t x, unsigned int y) {
    return x & ((1ULL << y) - 1ULL);
}
// 约束：无
class KernelEye {
public:
    __aicore__ inline KernelEye() {}
    __aicore__ inline void Init(GM_ADDR y, int totalMatrixNum, int numRows, int numColumns) {
        int formerNum = totalMatrixNum % GetBlockNum();
        int beginIndex = 0;
        if (GetBlockIdx() < formerNum) {
            this->MatrixNum = totalMatrixNum / GetBlockNum() + 1;
            beginIndex = this->MatrixNum * GetBlockIdx();
        } else {
            this->MatrixNum = totalMatrixNum / GetBlockNum();
            beginIndex = (this->MatrixNum + 1) * GetBlockIdx() - (GetBlockIdx() - formerNum);
        }
        lastTailNum = (64 - (beginIndex * numRows * numColumns * sizeof(DTYPE_Y)) % 64) % 64 / sizeof(DTYPE_Y);
        nextHeadNum = (64 - (beginIndex * numRows * numColumns + MatrixNum * numRows * numColumns) * sizeof(DTYPE_Y) % 64) % 64 / sizeof(DTYPE_Y);
        if (GetBlockIdx() == 0) {
            lastTailNum = 0;
        }
            // 末尾核心防止tenosr访问溢出
        if (GetBlockIdx() + 1 == GetBlockNum()) {
            nextHeadNum = 0;
        }
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + beginIndex * numRows * numColumns, MatrixNum * numRows * numColumns + nextHeadNum);
        this->numRows = numRows;
        this->numColumns = numColumns;
    }
    __aicore__ inline void Process() {
        int realTimes = min(numRows, numColumns);
        int index = 0;
        int size = numRows * numColumns;
        int i = 0;
        int inc = numColumns + 1;
        for (; i < MatrixNum; i++) {
            bool flag = false;
            int tmpIdx = index;
            int j = 0;
            // 处理前半部分
            for (; j < realTimes && tmpIdx < lastTailNum; j++) {
                tmpIdx += inc;
            }
            // 处理后半部分
            if (j < realTimes) {
                flag = true;
                for (; j < realTimes; j++) {
                    yGm.SetValue(tmpIdx, 1);
                    tmpIdx += inc;
                }
            }
            if (flag) {
                index += size;
                i++;
                break;
            }
            index += size;
        }
        for (; i < MatrixNum; i++) {
            int tmpIdx = index;
            for (int j = 0; j < realTimes; j++) {
                yGm.SetValue(tmpIdx, 1);
                tmpIdx += inc;
            }
            index += size;
        }
        int totalSize = MatrixNum * size;
        for (; i < MatrixNum + 64; i++) {
            int tmpIdx = index;
            for (int j = 0; j < realTimes; j++) {
                if (tmpIdx - totalSize >= nextHeadNum) {
                    return;
                }
                yGm.SetValue(tmpIdx, 1);
                tmpIdx += inc;
            }
            index += size;
        }
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECOUT, 1> outQueueY_aligned;
    TQue<QuePosition::VECOUT, 1> outQueueY_tail;
    GlobalTensor<DTYPE_Y> yGm;
    GlobalTensor<uint32_t> yGm_2;
    uint64_t mask[2];
    uint64_t mask_remain[2];
    int numRows;
    int numColumns;
    int MatrixNum;
    int lastTailNum;
    int nextHeadNum;
};
// 约束：sizeof(DTYPE_Y)*(tiling_data.numColumns+1)<32)
class KernelEye_slice {
public:
    __aicore__ inline KernelEye_slice() {}
    __aicore__ inline void Init(GM_ADDR y, int totalMatrixNum, int numRows, int numColumns, uint64_t mask0, uint64_t mask1, uint64_t mask_remain0,
                                uint64_t mask_remain1) {
        int formerNum = totalMatrixNum % GetBlockNum();
        signed int beginIndex = 0;
        if (GetBlockIdx() < formerNum) {
            this->MatrixNum = totalMatrixNum / GetBlockNum() + 1;
            beginIndex = this->MatrixNum * GetBlockIdx();
        } else {
            this->MatrixNum = totalMatrixNum / GetBlockNum();
            beginIndex = (this->MatrixNum + 1) * GetBlockIdx() - (GetBlockIdx() - formerNum);
        }
        if constexpr (std::is_same<DTYPE_Y, double>::value) {
                // 一个double相当于两个uint32_t
            yGm_2.SetGlobalBuffer((__gm__ uint32_t *)y + beginIndex * numRows * numColumns * 2, MatrixNum * numRows * numColumns * 2);
        } else {
            yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + beginIndex * numRows * numColumns, MatrixNum * numRows * numColumns);
        }
        this->numRows = numRows;
        this->numColumns = numColumns;
        this->mask[0] = mask0;
        this->mask[1] = mask1;
        this->mask_remain[0] = mask_remain0;
        this->mask_remain[1] = mask_remain1;
    }
    __aicore__ inline void ProcessBigMatrix_double() {
        int realTimes = min(numRows, numColumns);
        int index = 0;
        int DTYPE_SIZE = 4;
        uint32_t size = numRows * numColumns;
        uint32_t inc = numColumns + 1;
        uint32_t multiple = min((realTimes + 15) / 8 / 2, 300);
        uint32_t tilingSize = multiple * 256 / DTYPE_SIZE;
        AscendC::SliceInfo srcSliceInfo[] = {{0, tilingSize - 1, 0, 1, tilingSize}, {0, 0, 0, 1, 1}};
            // 间隔c个元素 迁移过去间距变成了22个元素，需要考虑当前32B，间隔就不是numColumns个元素了
        AscendC::SliceInfo dstSliceInfo[] = {
            {0, (multiple * 8 - 1) * (inc) * 2 + 1, (uint32_t)numColumns * 2 - (uint32_t)(32 / DTYPE_SIZE - 2), 1, (multiple * 8 - 1) * (inc) * 2 + 2},
            {0, 0, 0, 1, 1}};
        uint32_t repeatTimes = (realTimes - 1) / (8 * multiple);
        uint32_t remain = realTimes - repeatTimes * 8 * multiple;
            // 换算成uint32_t的剩余个数
        uint32_t remainSize = remain * 32 / DTYPE_SIZE;
        AscendC::SliceInfo srcSliceInfo_remain[] = {{0, remainSize - 1, 0, 1, remainSize}, {0, 0, 0, 1, 1}};
        // 间隔c个元素 迁移过去间距变成了22个元素，需要考虑当前32B，间隔就不是numColumns个元素了 间距多少个uint32_t
        AscendC::SliceInfo dstSliceInfo_remain[] = {
            {0, (remain - 1) * (inc) * 2 + 1, (uint32_t)numColumns * 2 - (uint32_t)(32 / DTYPE_SIZE - 2), 1, (remain - 1) * (inc) * 2 + 2}, {0, 0, 0, 1, 1}};
        pipe.InitBuffer(outQueueY_aligned, 1, 256 * multiple);
        pipe.InitBuffer(outQueueY_tail, 1, 256 * multiple);
        LocalTensor<uint32_t> yLocal_aligned = outQueueY_aligned.AllocTensor<uint32_t>();
        LocalTensor<uint32_t> yLocal_tail = outQueueY_tail.AllocTensor<uint32_t>();
        if (multiple == 0) {
            multiple = 1;
        }
        uint32_t t_m = tilingSize / multiple;
        Duplicate(yLocal_tail, (uint32_t)0, t_m, multiple, 1, 8);
        // 填充高32位
        Duplicate(yLocal_tail, (uint32_t)0x3FF00000, mask_remain, multiple, 1, 8);
        Duplicate(yLocal_aligned, (uint32_t)0, t_m, multiple, 1, 8);
        // 填充高32位
        Duplicate(yLocal_aligned, (uint32_t)0x3FF00000, mask, multiple, 1, 8);
        outQueueY_tail.EnQue<uint32_t>(yLocal_tail);
        yLocal_tail = outQueueY_tail.DeQue<uint32_t>();
        outQueueY_aligned.EnQue<uint32_t>(yLocal_aligned);
        yLocal_aligned = outQueueY_aligned.DeQue<uint32_t>();
        for (int i = 0; i < MatrixNum; i++) {
            int tmpIdx = index;
            for (int j = 0; j < repeatTimes; j++) {
                DataCopy(yGm_2[tmpIdx], yLocal_aligned, dstSliceInfo, srcSliceInfo);
                // 将double解析成uint32_t
                tmpIdx += multiple * 8 * inc * 2;
            }
            DataCopy(yGm_2[tmpIdx - (32 / DTYPE_SIZE - 2)], yLocal_tail, dstSliceInfo_remain, srcSliceInfo_remain);
            // 将double解析成uint32_t
            index += size * 2;
        }
        outQueueY_aligned.FreeTensor(yLocal_aligned);
        outQueueY_tail.FreeTensor(yLocal_tail);
    }
    __aicore__ inline void ProcessBigMatrix() {
        int realTimes = min(numRows, numColumns);
        int index = 0;
        uint32_t size = numRows * numColumns;
        uint32_t inc = numColumns + 1;
        uint32_t tilingSize = 256 / sizeof(DTYPE_Y);
        AscendC::SliceInfo srcSliceInfo[] = {{0, tilingSize - 1, 0, 1, tilingSize}, {0, 0, 0, 1, 1}};
        // 间隔c个元素 迁移过去间距变成了22个元素，需要考虑当前32B，间隔就不是numColumns个元素了
        AscendC::SliceInfo dstSliceInfo[] = {{0, 7 * (inc), (uint32_t)numColumns - (uint32_t)(32 / sizeof(DTYPE_Y) - 1), 1, 7 * (inc) + 1}, {0, 0, 0, 1, 1}};
        // 注意最后一个对角线元素不要使用该方法填充
        uint32_t repeatTimes = (realTimes - 1) / 8;
        uint32_t remain = realTimes - repeatTimes * 8;
        uint32_t remainSize = remain * 32 / sizeof(DTYPE_Y);
        AscendC::SliceInfo srcSliceInfo_remain[] = {{0, remainSize - 1, 0, 1, remainSize}, {0, 0, 0, 1, 1}};
        // 间隔c个元素 迁移过去间距变成了22个元素，需要考虑当前32B，间隔就不是numColumns个元素了
        AscendC::SliceInfo dstSliceInfo_remain[] = {
            {0, (remain - 1) * (inc), (uint32_t)numColumns - (uint32_t)(32 / sizeof(DTYPE_Y) - 1), 1, (remain - 1) * (inc) + 1}, {0, 0, 0, 1, 1}};
        pipe.InitBuffer(outQueueY_aligned, 1, 256);
        pipe.InitBuffer(outQueueY_tail, 1, 256);
        LocalTensor<DTYPE_Y> yLocal_aligned = outQueueY_aligned.AllocTensor<DTYPE_Y>();
        LocalTensor<DTYPE_Y> yLocal_tail = outQueueY_tail.AllocTensor<DTYPE_Y>();
        if constexpr (!std::is_same<DTYPE_Y, double>::value) {
            Duplicate(yLocal_aligned, (DTYPE_Y)0.0, tilingSize);
            Duplicate(yLocal_aligned, (DTYPE_Y)1.0, mask, 1, 1, 8);
            Duplicate(yLocal_tail, (DTYPE_Y)0.0, tilingSize);
            Duplicate(yLocal_tail, (DTYPE_Y)1.0, mask_remain, 1, 1, 8);
        }
        outQueueY_aligned.EnQue<DTYPE_Y>(yLocal_aligned);
        yLocal_aligned = outQueueY_aligned.DeQue<DTYPE_Y>();
        outQueueY_tail.EnQue<DTYPE_Y>(yLocal_tail);
        yLocal_tail = outQueueY_tail.DeQue<DTYPE_Y>();
        for (int i = 0; i < MatrixNum; i++) {
            int tmpIdx = index;
            if constexpr (!std::is_same<DTYPE_Y, double>::value) {
                for (int j = 0; j < repeatTimes; j++) {
                    DataCopy(yGm[tmpIdx], yLocal_aligned, dstSliceInfo, srcSliceInfo);
                    tmpIdx += 8 * inc;
                }
                DataCopy(yGm[tmpIdx - (32 / sizeof(DTYPE_Y) - 1)], yLocal_tail, dstSliceInfo_remain, srcSliceInfo_remain);
            }
            index += size;
        }
        outQueueY_aligned.FreeTensor(yLocal_aligned);
        outQueueY_tail.FreeTensor(yLocal_tail);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECOUT, 1> outQueueY_aligned;
    TQue<QuePosition::VECOUT, 1> outQueueY_tail;
    GlobalTensor<DTYPE_Y_REF> yGm;
    GlobalTensor<uint32_t> yGm_2;
    uint64_t mask[2];
    uint64_t mask_remain[2];
    int numRows;
    int numColumns;
    int MatrixNum;
    int lastTailNum;
    int nextHeadNum;
};
// 约束：列数是4的倍数 且 行数多于列数
class KernelEye_double_aligned {
public:
    __aicore__ inline KernelEye_double_aligned() {}
    __aicore__ inline void Init(GM_ADDR y, uint16_t totalMatrixNum, uint16_t numRows, uint16_t numColumns, TPipe *pipeIn) {
        pipe = pipeIn;
        uint16_t formerNum = totalMatrixNum % GetBlockNum();
        uint32_t beginIndex = 0;
        if (GetBlockIdx() < formerNum) {
            this->MatrixNum = totalMatrixNum / GetBlockNum() + 1;
            beginIndex = this->MatrixNum * GetBlockIdx();
        } else {
            this->MatrixNum = totalMatrixNum / GetBlockNum();
            beginIndex = this->MatrixNum * GetBlockIdx() + formerNum;
        }
        this->numRows = numRows;
        this->numColumns = numColumns;
        yGm.SetGlobalBuffer((__gm__ uint32_t *)y + beginIndex * numRows * numColumns * 2, MatrixNum * numRows * numColumns * 2);
    }
    __aicore__ inline void Process() {
        // 均是按int32为单位
        const uint32_t size = numColumns * numColumns * 2;
        const uint16_t inc = (numColumns + 1) * 2;
        const uint16_t oneCount = numColumns;
        pipe->InitBuffer(outQueueY, 1, 4 * size * MatrixNum);
        LocalTensor<uint32_t> yLocal = outQueueY.AllocTensor<uint32_t>();
        // 填0
        Duplicate(yLocal, (uint32_t)0, size);
        // 填1
        constexpr uint64_t magicMask = (uint64_t)144680345676153346;
        uint64_t mask[2] = {magicMask, 0};
        uint16_t groupSize = oneCount / 4;
        uint16_t repeatTimes = groupSize / 8;
        uint16_t remainSize = groupSize % 8;
        for (uint8_t j = 0; j < 4; j++) {
            uint64_t beginIndex = (j * inc) / 8 * 8;
            // 分成4组，一次8个
            for (uint16_t k = 0; k < repeatTimes; k++) {
                // 分组后，相邻对角线元素间隔块号为inc/2 uint32_t一块8个元素 间隔inc/2块 这里就不能再Duplicate中用repeatTimes参数，因为迭代间间隔大于255
                Duplicate(yLocal[beginIndex + k * 8 * 4 * inc], (uint32_t)0x3FF00000, mask, 1, inc / 2, 0);
            }
            // 分组后，相邻对角线元素间隔块号为inc uint32_t一块8个元素
            mask[0] = rotate_left_64(magicMask, ((j * inc) % 8));
            mask[0] = keep_low_y_bits(mask[0], 8 * remainSize);
            Duplicate(yLocal[beginIndex + repeatTimes * 8 * 4 * inc], (uint32_t)0x3FF00000, mask, 1, inc / 2, 0);
        }
        uint16_t i = 1;
        const uint16_t max_repeat_matrix_num = 255 * 64 / size;
        // copy
        for (; 2 * i < MatrixNum && i <= max_repeat_matrix_num; i *= 2) {
            Copy(yLocal[i * size], yLocal, 64, i * size / 64, {1, 1, 8, 8}); // i*size/64>255就溢出服了，
        }
        while ((MatrixNum - i) * size / 64 >= 256) {
            Copy(yLocal[i * size], yLocal, 64, max_repeat_matrix_num * size / 64, {1, 1, 8, 8}); // i*size/64>255就溢出服了，
            i += max_repeat_matrix_num;
        }
        Copy(yLocal[size * i], yLocal, 64, (MatrixNum - i) * size / 64, {1, 1, 8, 8});
        outQueueY.EnQue<uint32_t>(yLocal);
        yLocal = outQueueY.DeQue<uint32_t>();
        DataCopy(yGm, yLocal, {MatrixNum, static_cast<uint16_t>(size / 8), 0, static_cast<uint16_t>((numRows - numColumns) * numColumns / 4)});
    }

private:
    TPipe *pipe;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    GlobalTensor<uint32_t> yGm;
    uint16_t numRows;
    uint16_t numColumns;
    uint16_t MatrixNum;
};
extern "C" __global__ __aicore__ void eye_fp64(GM_ADDR y, GM_ADDR y_ref, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(1)) {
        GET_TILING_DATA(tiling_data, tiling);
        KernelEye op;
        op.Init(y, tiling_data.totalMatrixNum, tiling_data.numRows, tiling_data.numColumns);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        GET_TILING_DATA_WITH_STRUCT(EyeFp64TilingData_slice, tiling_data, tiling);
        KernelEye_slice op;
        op.Init(y, tiling_data.totalMatrixNum, tiling_data.numRows, tiling_data.numColumns, tiling_data.mask0, tiling_data.mask1, tiling_data.mask_remain0,
                tiling_data.mask_remain1);
        if constexpr (std::is_same<DTYPE_Y, double>::value) {
            op.ProcessBigMatrix_double();
        } else {
            op.ProcessBigMatrix();
        }
    } else if (TILING_KEY_IS(3)) {
        GET_TILING_DATA(tiling_data, tiling);
        KernelEye_double_aligned op;
        TPipe pipe;
        op.Init(y, tiling_data.totalMatrixNum, tiling_data.numRows, tiling_data.numColumns, &pipe);
        op.Process();
    }
}