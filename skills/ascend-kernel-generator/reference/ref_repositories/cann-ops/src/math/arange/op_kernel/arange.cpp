/*
 * Copyright (C) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file arange.cpp
 */
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

#define ALIGN_UP_32B_ELEMENTS(count, T) \
    (((((count) * sizeof(T)) + 31) / 32) * (32 / sizeof(T)))

template <typename TYPE_START, typename TYPE_STEP, typename TYPE_OUT>
class KernelArange
{
public:
  __aicore__ inline KernelArange() {}
  __aicore__ inline void Init(GM_ADDR start, GM_ADDR end, GM_ADDR step, GM_ADDR out,
                              uint32_t totalNum, uint32_t unitNum, uint32_t unitLoops, 
                              uint32_t tailNum)
  {
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");

    startGm.SetGlobalBuffer((__gm__ float *)start);
    stepGm.SetGlobalBuffer((__gm__ float *)step);
    outGm.SetGlobalBuffer((__gm__ float *)out, totalNum);

    pipe.InitBuffer(outQueue, BUFFER_NUM, unitNum * sizeof(float));
    pipe.InitBuffer(temp1, unitNum * sizeof(float));
    pipe.InitBuffer(temp2, unitNum * sizeof(float));
    pipe.InitBuffer(temp3, unitNum * sizeof(float));

    this->totalNum = totalNum;
    this->unitNum = unitNum;
    this->tailNum = tailNum;
    this->unitLoops = unitLoops;
  }

   __aicore__ inline void work_init()
  {
    TYPE_START start = startGm.GetValue(0);
    TYPE_STEP  step  = stepGm.GetValue(0);
    
    this->calc_init = temp1.Get<float>();
    this->calc_step = temp2.Get<float>();
    this->calc_temp = temp3.Get<float>();
    // AscendC::CreateVecIndex(calc_temp, (TYPE_OUT)0, this->unitNum);
    for (int32_t idx = 0; idx < this->unitNum; idx++)
    {
      calc_temp.SetValue(idx, (float)idx);
    }
    AscendC::Duplicate(calc_init, start, this->unitNum);
    AscendC::Duplicate(calc_step, step, this->unitNum);
    AscendC::Mul(calc_step, calc_step, calc_temp, this->unitNum);
    AscendC::Add(calc_init, calc_init, calc_step, this->unitNum);
    AscendC::Duplicate(calc_temp, float(0.0), this->unitNum);
    this->offset_step_base = this->unitNum * step;
    AscendC::Duplicate(calc_step, this->offset_step_base, this->unitNum);
  }

  __aicore__ inline void Process()
  {
    /*初始化第一个UNIT序列值*/
    work_init();

    for (int32_t i = 0; i < this->unitLoops; i++)
    {
      if( i == this->unitLoops -1
        && this->tailNum > 0 )
        {
          Compute(i, this->tailNum);
          CopyOut(i, this->tailNum);
        }
        else
        {
          Compute(i, this->unitNum);
          CopyOut(i, this->unitNum);
        }
    }
  }

private:

  __aicore__ inline void Compute(int32_t iter, int32_t num)
  {
    uint32_t calc_num = ALIGN_UP_32B_ELEMENTS(num, float);
    AscendC::LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
    AscendC::Add(outLocal, this->calc_init, this->calc_temp, calc_num);
    AscendC::Add(this->calc_temp, this->calc_temp, this->calc_step, calc_num);
    outQueue.EnQue<float>(outLocal);

  }

  __aicore__ inline void CopyOut(int32_t iter, int32_t num)
  {
      uint32_t copy_num = ALIGN_UP_32B_ELEMENTS(num, float);
      AscendC::LocalTensor<float> outLocal = outQueue.DeQue<float>();
      // AscendC::DumpTensor(outLocal,iter, num);
      AscendC::DataCopy(outGm[iter*this->unitNum], outLocal, copy_num);
      outQueue.FreeTensor(outLocal);

  }
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueue;
  AscendC::TBuf<AscendC::QuePosition::VECCALC> temp1, temp2, temp3;
  AscendC::GlobalTensor<float> startGm;
  AscendC::GlobalTensor<float> stepGm;
  AscendC::GlobalTensor<float> outGm;
  AscendC::LocalTensor<float> calc_init, calc_step, calc_temp;

  uint32_t totalNum;
  uint32_t unitNum;
  uint32_t unitLoops;
  uint32_t tailNum;
  /*UNIT之间元素值差间隔*/
  float offset_step_base;
};

/*INT64/BF16/FP16均转成FP32运算*/
template <typename TYPE_START, typename TYPE_STEP, typename TYPE_OUT>
class KernelArange_Cast
{
public:
  __aicore__ inline KernelArange_Cast() {}
  __aicore__ inline void Init(GM_ADDR start, GM_ADDR end, GM_ADDR step, GM_ADDR out,
                              uint32_t totalNum, uint32_t unitNum, uint32_t unitLoops, 
                              uint32_t tailNum)
  {
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");

    startGm.SetGlobalBuffer((__gm__ TYPE_START *)start);
    stepGm.SetGlobalBuffer((__gm__ TYPE_STEP *)step);
    outGm.SetGlobalBuffer((__gm__ TYPE_OUT *)out, totalNum);

    pipe.InitBuffer(inQueue, BUFFER_NUM, sizeof(TYPE_START));
    pipe.InitBuffer(outQueue, BUFFER_NUM, unitNum * sizeof(TYPE_OUT));
    pipe.InitBuffer(temp1, unitNum * sizeof(float));
    pipe.InitBuffer(temp2, unitNum * sizeof(float));
    pipe.InitBuffer(temp3, unitNum * sizeof(float));
    pipe.InitBuffer(temp4, unitNum * sizeof(float));
    pipe.InitBuffer(tempFloat, 2*sizeof(float)); // 2表示两个float类型的大小

    this->totalNum = totalNum;
    this->unitNum = unitNum;
    this->tailNum = tailNum;
    this->unitLoops = unitLoops;
  }

   __aicore__ inline void work_init()
  {
    AscendC::LocalTensor<TYPE_START> startLocal_in = inQueue.AllocTensor<TYPE_START>();
    AscendC::DataCopy(startLocal_in, startGm, ALIGN_UP_32B_ELEMENTS(1, TYPE_START));
    inQueue.EnQue<TYPE_START>(startLocal_in);
    AscendC::LocalTensor<TYPE_START> startLocal_out = inQueue.DeQue<TYPE_START>();
    AscendC::LocalTensor<float> float_start_tensor = tempFloat.Get<float>(0);
    if constexpr ( std::is_same_v<TYPE_START, bfloat16_t>
                  || std::is_same_v<TYPE_START, half> )
    {
      AscendC::Cast(float_start_tensor, startLocal_out, AscendC::RoundMode::CAST_NONE,
                    ALIGN_UP_32B_ELEMENTS(1, TYPE_START));
    }
    else
    {
      AscendC::Cast(float_start_tensor, startLocal_out, AscendC::RoundMode::CAST_ROUND,
                    ALIGN_UP_32B_ELEMENTS(1, TYPE_START));
    }
    float float_start = float_start_tensor.GetValue(0);
    inQueue.FreeTensor(startLocal_in);

    AscendC::LocalTensor<TYPE_STEP> stepLocal_in = inQueue.AllocTensor<TYPE_STEP>();
    AscendC::DataCopy(stepLocal_in, stepGm, ALIGN_UP_32B_ELEMENTS(1, TYPE_STEP));
    inQueue.EnQue<TYPE_STEP>(stepLocal_in);
    AscendC::LocalTensor<TYPE_STEP> stepLocal_out = inQueue.DeQue<TYPE_STEP>();
    AscendC::LocalTensor<float> float_step_tensor = tempFloat.Get<float>(1);;
    if constexpr ( std::is_same_v<TYPE_STEP, bfloat16_t>
                  || std::is_same_v<TYPE_STEP, half> )
    {
      AscendC::Cast(float_step_tensor, stepLocal_out, AscendC::RoundMode::CAST_NONE,
                    ALIGN_UP_32B_ELEMENTS(1, TYPE_STEP));
    }
    else
    {
      AscendC::Cast(float_step_tensor, stepLocal_out, AscendC::RoundMode::CAST_ROUND,
                    ALIGN_UP_32B_ELEMENTS(1, TYPE_STEP));
    }
    float float_step = float_step_tensor.GetValue(0);
    inQueue.FreeTensor(stepLocal_in);

    this->calc_init = temp1.Get<float>();
    this->calc_step = temp2.Get<float>();
    this->calc_temp = temp3.Get<float>();
    this->calc_out  = temp4.Get<float>();
    // AscendC::CreateVecIndex(calc_temp, (TYPE_OUT)0, this->unitNum);
    for (int32_t idx = 0; idx < this->unitNum; idx++)
    {
      calc_temp.SetValue(idx, (float)idx);
    }
    AscendC::Duplicate(calc_init, float_start, this->unitNum);
    AscendC::Duplicate(calc_step, float_step, this->unitNum);
    AscendC::Mul(calc_step, calc_step, calc_temp, this->unitNum);
    AscendC::Add(calc_init, calc_init, calc_step, this->unitNum);
    AscendC::Duplicate(calc_temp, (float)0.0, this->unitNum);
    this->offset_step_base = this->unitNum * float_step;
    AscendC::Duplicate(calc_step, this->offset_step_base, this->unitNum);
  }

  __aicore__ inline void Process()
  {
    /*初始化第一个UNIT序列值*/
    work_init();

    for (int32_t i = 0; i < this->unitLoops; i++)
    {
      if( i == this->unitLoops -1
        && this->tailNum > 0 )
        {
          Compute(i, this->tailNum);
          CopyOut(i, this->tailNum);
        }
        else
        {
          Compute(i, this->unitNum);
          CopyOut(i, this->unitNum);
        }
    }
  }

private:

  __aicore__ inline void Compute(int32_t iter, int32_t num)
  {
    uint32_t calc_num = ALIGN_UP_32B_ELEMENTS(num, float);
    
    AscendC::LocalTensor<TYPE_OUT> outLocal = outQueue.AllocTensor<TYPE_OUT>();
    AscendC::Add(this->calc_out, this->calc_init, this->calc_temp, calc_num);
    AscendC::Add(this->calc_temp, this->calc_temp, this->calc_step, calc_num);
    AscendC::Cast(outLocal, this->calc_out, AscendC::RoundMode::CAST_ROUND, calc_num);
    outQueue.EnQue<TYPE_OUT>(outLocal);
  }

  __aicore__ inline void CopyOut(int32_t iter, int32_t num)
  {
      uint32_t copy_num = ALIGN_UP_32B_ELEMENTS(num, TYPE_OUT);

      AscendC::LocalTensor<TYPE_OUT> outLocal = outQueue.DeQue<TYPE_OUT>();
      // AscendC::DumpTensor(outLocal,iter, num);
      AscendC::DataCopy(outGm[iter*this->unitNum], outLocal, copy_num);
      outQueue.FreeTensor(outLocal);
  }

  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueue;
  AscendC::TBuf<AscendC::QuePosition::VECIN> tempFloat;
  AscendC::TBuf<AscendC::QuePosition::VECCALC> temp1, temp2, temp3, temp4;
  AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueue;
  AscendC::GlobalTensor<TYPE_START> startGm;
  AscendC::GlobalTensor<TYPE_STEP> stepGm;
  AscendC::GlobalTensor<TYPE_OUT> outGm;
  AscendC::LocalTensor<float> calc_init, calc_step, calc_temp;
  AscendC::LocalTensor<float> calc_out;

  uint32_t totalNum;
  uint32_t unitNum;
  uint32_t unitLoops;
  uint32_t tailNum;
  /*UNIT之间元素值差间隔*/
  float offset_step_base;
};

extern "C" __global__ __aicore__ void arange(GM_ADDR start, GM_ADDR end, GM_ADDR step, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
  GET_TILING_DATA(tiling_data, tiling);

  if(TILING_KEY_IS(1)) {
    KernelArange<float, float, float> op;
    
    op.Init(start, end, step, out,
            tiling_data.totalNum,
            tiling_data.unitNum,
            tiling_data.unitLoops,
            tiling_data.tailNum);

    op.Process();
  } else if(TILING_KEY_IS(0)) {
    KernelArange_Cast<DTYPE_START, DTYPE_STEP, DTYPE_OUT> op;
    
    op.Init(start, end, step, out,
            tiling_data.totalNum,
            tiling_data.unitNum,
            tiling_data.unitLoops,
            tiling_data.tailNum);

    op.Process();
  }
}

#ifndef ASCENDC_CPU_DEBUG
// call of kernel function
void arange_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *start, uint8_t *end, uint8_t *step,
               uint8_t *out, uint8_t *workspace, uint8_t *tiling)
{
  arange<<<blockDim, l2ctrl, stream>>>(start, end, step, out, workspace, tiling);
}
#endif