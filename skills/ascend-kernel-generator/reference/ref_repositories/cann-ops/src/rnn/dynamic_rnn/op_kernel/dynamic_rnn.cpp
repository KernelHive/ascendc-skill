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
 * \file dynamic_rnn_910b.cpp
 * \brief
 */
#include "LstmFP16.cpp"
#include "LstmFP32.cpp"

extern "C" __global__ __aicore__ void dynamic_rnn(GM_ADDR inputX, GM_ADDR weight, GM_ADDR bias, GM_ADDR seqLength,
                                                  GM_ADDR initH, GM_ADDR initC, GM_ADDR wCi, GM_ADDR wCf, GM_ADDR wCo,
                                                  GM_ADDR mask, GM_ADDR outputY, GM_ADDR outputH, GM_ADDR outputC,
                                                  GM_ADDR outputI, GM_ADDR outputJ, GM_ADDR outputF, GM_ADDR outputO,
                                                  GM_ADDR outputTanhC, GM_ADDR workspace, GM_ADDR rnnTiling) {
  set_mask_norm();
  GET_TILING_DATA(tiling_data, rnnTiling);
  const DynamicRNNTilingData* __restrict tilingData = &tiling_data;
  const TCubeTiling* __restrict inputMMTiling = &(tilingData->inputMMParam);
  const TCubeTiling* __restrict hiddenMMTiling = &(tilingData->hiddenMMParam);

  LstmBean lstmBean;
  lstmBean.inputX = inputX;
  lstmBean.weight = weight;
  lstmBean.bias = bias;
  lstmBean.seqLength = seqLength;
  lstmBean.initH = initH;
  lstmBean.initC = initC;
  lstmBean.wCi = wCi;
  lstmBean.wCf = wCf;
  lstmBean.wCo = wCo;
  lstmBean.mask = mask;
  lstmBean.outputY = outputY;
  lstmBean.outputH = outputH;
  lstmBean.outputC = outputC;
  lstmBean.outputI = outputI;
  lstmBean.outputJ = outputJ;
  lstmBean.outputF = outputF;
  lstmBean.outputO = outputO;
  lstmBean.outputTanhC = outputTanhC;

  if (TILING_KEY_IS(10000001)) {
    LstmMmSplitNDNDFP16<half> lstmOp;
    REGIST_MATMUL_OBJ(&lstmOp.pipe, GetSysWorkSpacePtr(), lstmOp.inputMM, inputMMTiling, lstmOp.hiddenMM, hiddenMMTiling);

    lstmOp.Init(inputX, weight, bias, seqLength, initH, initC, wCi, wCf, wCo, mask, outputY, outputH, outputC, outputI,
                outputJ, outputF, outputO, outputTanhC, &tiling_data, workspace);
    lstmOp.Process();
  } else if (TILING_KEY_IS(10000002)) {
    LstmMmSplitNDNDFP32<float> lstmOp;
    REGIST_MATMUL_OBJ(&lstmOp.pipe, GetSysWorkSpacePtr(), lstmOp.inputMM, inputMMTiling, lstmOp.hiddenMM, hiddenMMTiling);

    lstmOp.Init(inputX, weight, bias, seqLength, initH, initC, wCi, wCf, wCo, mask, outputY, outputH, outputC, outputI,
                outputJ, outputF, outputO, outputTanhC, &tiling_data, workspace);
    lstmOp.Process();
  } else if (TILING_KEY_IS(10000004)) {
    LstmMmSplitNDNDFP16<bfloat16_t> lstmOp;
    REGIST_MATMUL_OBJ(&lstmOp.pipe, GetSysWorkSpacePtr(), lstmOp.inputMM, inputMMTiling, lstmOp.hiddenMM, hiddenMMTiling);

    lstmOp.Init(inputX, weight, bias, seqLength, initH, initC, wCi, wCf, wCo, mask, outputY, outputH, outputC, outputI,
                outputJ, outputF, outputO, outputTanhC, &tiling_data, workspace);
    lstmOp.Process();
  }
}
