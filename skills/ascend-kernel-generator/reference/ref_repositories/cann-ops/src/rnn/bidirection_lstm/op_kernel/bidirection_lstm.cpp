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
 * @file bidirection_lstm.cpp
 */

#include "lstm_bidir_fp16.cpp"

extern "C" __global__ __aicore__ void bidirection_lstm(GM_ADDR x, GM_ADDR init_h, GM_ADDR init_c, GM_ADDR w_ih, GM_ADDR w_hh,
                                                  GM_ADDR b_ih, GM_ADDR b_hh, GM_ADDR w_ih_reverse, GM_ADDR w_hh_reverse,
                                                  GM_ADDR b_ih_reverse, GM_ADDR b_hh_reverse, GM_ADDR y, GM_ADDR output_h,
                                                  GM_ADDR output_c, GM_ADDR usrworkspace, GM_ADDR lstmTiling) {
  SetAtomicNone();
  GET_TILING_DATA(tiling_data, lstmTiling);
  const BidirectionLSTMTilingData* __restrict tilingData = &tiling_data;
  if (TILING_KEY_IS(10000001)) {
    LstmBidirFP16 lstm_handle;
    lstm_handle.Init(x, init_h, init_c, w_ih, w_hh, b_ih, b_hh, w_ih_reverse, w_hh_reverse,
                    b_ih_reverse, b_hh_reverse, nullptr, y, output_h, output_c, usrworkspace, tilingData);
    lstm_handle.Process();
  }
}
