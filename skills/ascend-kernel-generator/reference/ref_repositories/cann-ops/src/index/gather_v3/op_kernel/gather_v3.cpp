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
 * \file gather_v3.cpp
 * \brief
 */
#include "gather_v3_tmpl_0.h"
#include "gather_v3_tmpl_1.h"
#include "gather_v3_tmpl_2.h"
#include "gather_v3_tmpl_3.h"

using namespace GatherV3;

#define PROC_TILING_KEY(TMPL, KEY) do {\
    if (TILING_KEY_IS(KEY)) {\
        GatherV3Tmpl##TMPL <DTYPE_X, DTYPE_INDICES> op(&tilingData);\
        op.Init(x, indices, y, &pipe);\
        op.Process<GatherV3Tmpl##TMPL <DTYPE_X, DTYPE_INDICES>,\
                  &GatherV3Tmpl##TMPL <DTYPE_X, DTYPE_INDICES>::ProcessSingleTile##KEY >(&op);\
        return;\
    }\
} while(0)

extern "C" __global__ __aicore__ void gather_v3(GM_ADDR x, GM_ADDR indices, GM_ADDR axis, GM_ADDR y, GM_ADDR workspace,
                                                       GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    TPipe pipe;

    PROC_TILING_KEY(0, 10430);
    PROC_TILING_KEY(0, 10330);
    PROC_TILING_KEY(0, 10200);
    PROC_TILING_KEY(0, 12330);
    PROC_TILING_KEY(0, 11100);
    PROC_TILING_KEY(1, 20330);
    PROC_TILING_KEY(1, 20320);
    PROC_TILING_KEY(1, 20310);
    PROC_TILING_KEY(1, 20230);
    PROC_TILING_KEY(1, 20220); 
    PROC_TILING_KEY(1, 20210);
    PROC_TILING_KEY(1, 20130);
    PROC_TILING_KEY(1, 20131);
    PROC_TILING_KEY(1, 20132);
    PROC_TILING_KEY(1, 20120);
    PROC_TILING_KEY(1, 20110);
    PROC_TILING_KEY(1, 30330);
    PROC_TILING_KEY(1, 30320);
    PROC_TILING_KEY(1, 30310);
    PROC_TILING_KEY(1, 30230);
    PROC_TILING_KEY(1, 30220); 
    PROC_TILING_KEY(1, 30210);
    PROC_TILING_KEY(1, 30130);  
    PROC_TILING_KEY(1, 30131);   
    PROC_TILING_KEY(1, 30120);
    PROC_TILING_KEY(1, 30110);
    PROC_TILING_KEY(1, 40330);
    PROC_TILING_KEY(1, 40331);  
    PROC_TILING_KEY(1, 40320);
    PROC_TILING_KEY(1, 40310);
    PROC_TILING_KEY(1, 50330);    
    PROC_TILING_KEY(1, 50320);
    PROC_TILING_KEY(1, 50310);
    PROC_TILING_KEY(2, 60020);
    PROC_TILING_KEY(2, 60010);    
    PROC_TILING_KEY(2, 70330);
    PROC_TILING_KEY(2, 70320);
    PROC_TILING_KEY(2, 70310);
    PROC_TILING_KEY(2, 70230);
    PROC_TILING_KEY(2, 70220); 
    PROC_TILING_KEY(2, 70210);
    PROC_TILING_KEY(2, 70130);    
    PROC_TILING_KEY(2, 70131);    
    PROC_TILING_KEY(2, 70120);
    PROC_TILING_KEY(2, 70110);
    PROC_TILING_KEY(2, 70001);
    PROC_TILING_KEY(3, 80330);
    PROC_TILING_KEY(3, 80220);
}