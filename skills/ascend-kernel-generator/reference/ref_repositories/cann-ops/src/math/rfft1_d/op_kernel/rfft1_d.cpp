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
 * \file rfft1_d.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matrix/matmul/matmul.h"
#include "lib/pad/kernel_operator_pad_intf.h"
#include "lib/matmul_intf.h"
#include "rfft1_d_colley_tukey.h"
#include "rfft1_d_bluestein.h"
#include "rfft1_d.h"

static const uint32_t DFT_BORDER_VALUE = 4096;

extern "C" __global__ __aicore__ void rfft1_d(GM_ADDR x, GM_ADDR dft, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);
  
    GET_TILING_DATA(tilingData, tiling);
  
    if (tilingData.length <= DFT_BORDER_VALUE)  // FastDFT for length <= 4096
    {
        uint32_t* factorsP = const_cast<uint32_t*>(tilingData.factors);
    
        KernelRfftFastDFT op(tilingData.length, tilingData.batchesPerCore, tilingData.leftOverBatches,
                             tilingData.normal, tilingData.dftRealOverallSize, factorsP);
    
        uint32_t modeLength = 2 * (tilingData.length / 2 + 1);
    
        auto t1 = PrepareTiling((op.batches + op.advancedBatches) / GetBlockNum(), op.modeLength, tilingData.length);
        REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.matmulObj, (void*)&t1, op.matmulObjNZ, (void*)&t1);
        op.Init(x, dft, y, userWorkspace);
        op.Process();
    } else if (!tilingData.isBluestein) // Cooley-Tukey algorithm for long seqs
    {
        KernelRfftCooleyTukey op(&tilingData);
    
        auto t1 = PrepareTiling1(tilingData.length / tilingData.factors[0], tilingData.factors[0] * 2, tilingData.factors[0]);
        auto t2 = PrepareTiling2(tilingData.factors[1], tilingData.factors[0] * 2, tilingData.factors[1]);
        auto t3 = PrepareTiling3(tilingData.factors[2] / 2, (tilingData.length / tilingData.factors[2]) * 2,
                                 tilingData.factors[2]);
    
        op.Init(x, dft, y, userWorkspace);
        REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.matmulObj1, (void*)&t1, op.matmulObj2, (void*)&t2,
                          op.matmulObj3, (void*)&t3);
    
        op.Process();
    } else // Bluestein algorithm for other cases
    { 
        KernelRfftBluestein op(&tilingData);
    
        auto t1 = PrepareTiling4(tilingData.lengthPad / tilingData.factors[0], tilingData.factors[0] * 2, 2 * tilingData.factors[0]);
        auto t2 = PrepareTiling5(tilingData.factors[1], tilingData.factors[0] * 2, tilingData.factors[1]);
        auto t3 = PrepareTiling6(tilingData.factors[2], (tilingData.lengthPad / tilingData.factors[2]) * 2, tilingData.factors[2]);
        auto t4 = PrepareTiling7(tilingData.length, 2, 1);
    
        op.Init(x, dft, y, userWorkspace);
        REGIST_MATMUL_OBJ(&op.ct.pipe, GetSysWorkSpacePtr(), op.ct.matmulObj1, (void *)&t1, op.ct.matmulObj2, (void *)&t2, op.ct.matmulObj3, (void *)&t3, op.ct.matmulObj4, (void *) &t4);
    
        op.Process();
    }
    return;
}