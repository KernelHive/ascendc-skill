/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelEye {
public:
    __aicore__ inline KernelEye() {}
    __aicore__ inline void Init(GM_ADDR y, GM_ADDR y_ref,
                                uint32_t typeKey, uint32_t blockLength, uint32_t tileNum,
                                uint32_t tileLength, uint32_t lasttileLength, int32_t num_columns,
                                int32_t num_rows, int32_t dtype,
                                int32_t mark, int32_t batchSize, int32_t batchNum) {

        this->batchSize = batchSize;
        this->batchNum = batchNum;
        this->mark = mark;
        this->num_rows = num_rows;
        this->num_columns = num_columns;
        this->dtype = dtype;
        this->blockLength = blockLength;
        this->tileNum = tileNum;
        this->tileLength = tileLength / BUFFER_NUM;
        this->lasttileLength = lasttileLength / BUFFER_NUM;
        this->typeKey = typeKey;

        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y_ref + this->blockLength * GetBlockIdx(), this->blockLength);
    }
    __aicore__ inline void Process() {
        int32_t index, t;
        if(mark == 0){
            for(int32_t i = 0; i < num_rows; i++){
                index = i * num_columns + i;
                if(i < num_columns){
                    yGm.SetValue(index, 1);
                }
            }
        }else{
            for(int32_t i = 0; i < batchNum; i++){
                for(int32_t j = 0; j < num_rows; j++){
                    if(j < num_columns){
                        t = j * num_columns + j;
                        index = i * batchSize + t;
                        yGm.SetValue(index, 1);
                    }
                }
            }
        }
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> tmpBuf1, tmpBuf2, tmpBuf3;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueIN;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueOUT;
    GlobalTensor<DTYPE_Y> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t lasttileLength;
    uint32_t typeKey;
    int32_t *batch_shape;
    int32_t dtype;
    int32_t num_columns;
    int32_t num_rows;
    int32_t mark;
    int32_t batchSize, batchNum;
};

extern "C" __global__ __aicore__ void eye(GM_ADDR y, GM_ADDR y_ref, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    KernelEye op;

    op.Init(y, y_ref, tiling_data.typeKey, tiling_data.blockLength,
            tiling_data.tileNum, tiling_data.tileLength, tiling_data.lasttileLength,
            tiling_data.num_columns, tiling_data.num_rows,
            tiling_data.dtype, tiling_data.mark, tiling_data.batchSize, tiling_data.batchNum);
    op.Process();
}
