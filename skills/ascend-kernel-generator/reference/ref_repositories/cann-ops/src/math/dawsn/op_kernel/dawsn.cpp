/*
 * @file dawsn.cpp
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2; 

struct TilingParam {
    uint32_t totalLength;
    uint32_t tileNum;
    uint32_t ALIGN_NUM;
    uint32_t tiling_size;
    uint32_t block_size;
    uint32_t aivNum;
    uint32_t core_size;
    uint32_t core_remain;
};

template<typename TYPE_X, typename TYPE_Y> class Dawsn {
    static constexpr float HALF_FACTOR = 0.5;

    static constexpr float COEF_AN[] = {
               1.13681498971755972054E-11,
               8.49262267667473811108E-10,
               1.94434204175553054283E-8,
               9.53151741254484363489E-7,
               3.07828309874913200438E-6,
               3.52513368520288738649E-4,
               -8.50149846724410912031E-4,
               4.22618223005546594270E-2,
               -9.17480371773452345351E-2,
               9.99999999999999994612E-1};   

    static constexpr float COEF_AN_COUNT = 9;

    static constexpr float COEF_AD[] = {
               2.40372073066762605484E-11,
               1.48864681368493396752E-9,
               5.21265281010541664570E-8,
               1.27258478273186970203E-6,
               2.32490249820789513991E-5,
               3.25524741826057911661E-4,
               3.48805814657162590916E-3,
               2.79448531198828973716E-2,
               1.58874241960120565368E-1,
               5.74918629489320327824E-1,
               1.00000000000000000539E0};

    static constexpr float COEF_AD_COUNT = 10;

    static constexpr float COEF_BN[] = {5.08955156417900903354E-1,
               -2.44754418142697847934E-1,
               9.41512335303534411857E-2,
               -2.18711255142039025206E-2,
               3.66207612329569181322E-3,
               -4.23209114460388756528E-4,
               3.59641304793896631888E-5,
               -2.14640351719968974225E-6,
               9.10010780076391431042E-8,
               -2.40274520828250956942E-9,
               3.59233385440928410398E-11};

    static constexpr float COEF_BN_COUNT = 10;

    static constexpr float COEF_BD[] = {-6.31839869873368190192E-1,
               2.36706788228248691528E-1,
               -5.31806367003223277662E-2,
               8.48041718586295374409E-3,
               -9.47996768486665330168E-4,
               7.81025592944552338085E-5,
               -4.55875153252442634831E-6,
               1.89100358111421846170E-7,
               -4.91324691331920606875E-9,
               7.18466403235734541950E-11};

    static constexpr float COEF_BD_COUNT = 10;

    static constexpr float COEF_CN[] = {-5.90592860534773254987E-1,
               6.29235242724368800674E-1,
               -1.72858975380388136411E-1,
               1.64837047825189632310E-2,
               -4.86827613020462700845E-4};

    static constexpr float COEF_CN_COUNT = 4;

    static constexpr float COEF_CD[] = {-2.69820057197544900361E0,
               1.73270799045947845857E0,
               -3.93708582281939493482E-1,
               3.44278924041233391079E-2,
               -9.73655226040941223894E-4};

    static constexpr float COEF_CD_COUNT = 5;
    static constexpr float THRESHOLD_3_25 = 3.25;
    static constexpr float THRESHOLD_6_25 = 6.25;
    static constexpr float THRESHOLD_1E_9 = 1.0e9;

public:
    __aicore__ inline Dawsn() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, TilingParam& paramList) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        this->totalLength = paramList.totalLength;
        uint32_t remainLength = GetBlockNum() == GetBlockIdx() + 1 ? paramList.core_remain : 0;
        this->blockLength = paramList.core_size + remainLength;
        this->tileLength = paramList.block_size;
        uint32_t ALIGN_NUM = paramList.ALIGN_NUM
        ASSERT(ALIGN_NUM != 0 && "ALIGN_NUM can not be zero!");
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);
        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        // get start index for current core, core parallel
        auto startPointer = paramList.core_size * GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + startPointer, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_X*)y + startPointer, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
    }

    __aicore__ inline void Process() {
        for (int32_t i = 0; i < this->totalLength; i++) {
          float x = xGm.GetValue(i);
          float res = 0;
          int sign = (x < 0) ? -1 : 1;
          x = (x < 0) ? -x : x;

          if (x < THRESHOLD_3_25) {
              res = _calc_condition_lt_three_p_two_five(x) * sign;
              yGm.SetValue(i, res);
          } else if (x < THRESHOLD_6_25) {
              res = _calc_condition_lt_six_p_two_five(x) * sign;
              yGm.SetValue(i, res);
          } else if (x <= THRESHOLD_1E_9) {
              res = _calc_condition_le_one_e_nine(x) * sign;
              yGm.SetValue(i, res);
          } else {
              res = _calc_condition_gt_one_e_nine(x) * sign;
              yGm.SetValue(i, res);
          }
        }
    }

    __aicore__ inline float _polevl(float data_x, const float *coef, int N) {
      float res = coef[0];
      for (int i = 1; i <= N; ++i) {
          res = res * data_x + coef[i];
      }
      return res;
    }

    __aicore__ inline float _p1evl(float data_x, const float *coef, int N) {
      float res = coef[0] + data_x;
      for (int i = 1; i < N; ++i) {
          res = res * data_x + coef[i];
      }
      return res;
    }

    __aicore__ inline float _calc_condition_lt_three_p_two_five(float input_x) {
      float x_square = input_x * input_x;
      float polevl_an = _polevl(x_square, COEF_AN, COEF_AN_COUNT);
      float polevl_ad = _polevl(x_square, COEF_AD, COEF_AD_COUNT);
      ASSERT(polevl_ad != 0 && "polevl_ad can not be zero!");
      float res = input_x * polevl_an / polevl_ad;
      return res;
    }

    __aicore__ inline float _calc_condition_lt_six_p_two_five(float input_x) {
      ASSERT(input_x != 0 && "input_x can not be zero!");
      float temp = static_cast<float>(1.0) / (input_x * input_x);
      float rec =  static_cast<float>(1.0) / input_x;
      float polevl_bn = _polevl(temp, COEF_BN, COEF_BN_COUNT) * temp;
      float p1evl_bd = _p1evl(temp, COEF_BD, COEF_BD_COUNT) * input_x;
      float res = (rec + polevl_bn / p1evl_bd) * HALF_FACTOR;
      return res;
    }

    __aicore__ inline float _calc_condition_le_one_e_nine(float x) {
      ASSERT(x != 0 && "x can not be zero!");
      float temp = static_cast<float>(1.0) / (x * x);
      float rec = static_cast<float>(1.0) / x;
      float polevl_cn = _polevl(temp, COEF_CN, COEF_CN_COUNT) * temp;
      float p1evl_cd = _p1evl(temp, COEF_CD, COEF_CD_COUNT) * x;
      float res = (rec + polevl_cn / p1evl_cd) * HALF_FACTOR;
      return res;
    }

    __aicore__ inline float _calc_condition_gt_one_e_nine(float input_x) {
      ASSERT(input_x != 0 && "input_x can not be zero!");
      float res = HALF_FACTOR / input_x;
      return res;
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_X> yGm;
    
    DTYPE_X res;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t totalLength;
};

extern "C" __global__ __aicore__ void dawsn(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TilingParam paramList = {
      .totalLength = tiling_data.totalLength,
      .tileNum = tiling_data.tileNum,
      .ALIGN_NUM = tiling_data.ALIGN_NUM,
      .tiling_size = tiling_data.tiling_size,
      .block_size = tiling_data.block_size,
      .aivNum = tiling_data.aivNum,
      .core_size = tiling_data.core_size,
      .core_remain = tiling_data.core_remain
    };

    Dawsn<DTYPE_X, DTYPE_Y> op;
    op.Init(x, y, paramList);
    op.Process();
}