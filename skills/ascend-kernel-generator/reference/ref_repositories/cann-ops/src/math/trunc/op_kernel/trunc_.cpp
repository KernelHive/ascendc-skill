/**
 * Copyright (C) Henan KunLun Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */
#include "sync_kl.h"
constexpr uint32_t INPUT_X_LW = 1;
constexpr uint32_t OUTPUT_Y_LW = 1;
using namespace AscendC;
template<typename TYPE_INPUT_X, typename TYPE_OUTPUT_Y> class KernelTrunc {
public:
    __aicore__ inline KernelTrunc() {}
    __aicore__ inline void Init(TPipe* pipe, 
                            GM_ADDR input_xAddr, GM_ADDR output_yAddr // IO的GM地址
                            , uint32_t& formerLen, uint32_t& formerNum, uint32_t& tailLen, 
                            uint32_t& totalLen // 核内切分参数
                        )
    {
         // 初始化算子属性
        this->formerLen=formerLen;
        this->formerNum=formerNum;
        this->tailLen=tailLen;
         // 核内切分参数
         // 初始化GM Tensor 与 TQue
        input_xGM.SetGlobalBuffer((__gm__ TYPE_INPUT_X*)input_xAddr, totalLen*INPUT_X_LW);
        output_yGM.SetGlobalBuffer((__gm__ TYPE_OUTPUT_Y*)output_yAddr, totalLen*OUTPUT_Y_LW);
        
        pipe->InitBuffer(input_xQue, 2, formerLen*INPUT_X_LW*sizeof(TYPE_INPUT_X));
        if constexpr(std::is_same_v<TYPE_INPUT_X, bfloat16_t>){
            pipe->InitBuffer(output_yQue, 2, formerLen*OUTPUT_Y_LW*4);
        }else{
            pipe->InitBuffer(output_yQue, 2, formerLen*OUTPUT_Y_LW*sizeof(TYPE_OUTPUT_Y));
        }
    }

    __aicore__ inline void Process()
    {
         // 读取变量
        const uint32_t formerNum = this->formerNum, formerLen = this->formerLen, tailLen = this->tailLen;
        {
            uint32_t t = 0;
            for (; t < formerNum; ++t){  // 遍历大Tiling
                ProcessTrunc(
                    t*formerLen,
                    formerLen
                );
            }
            if(tailLen) {  // 遍历小Tiling
                ProcessTrunc(
                    t*formerLen,
                    tailLen
                );
            }
        }
    }

private:
    __aicore__ inline void ProcessTrunc(const uint32_t& _offset_, const uint32_t& _len_){
         // CopyIn
        LocalTensor<TYPE_INPUT_X> input_x = input_xQue.AllocTensor<TYPE_INPUT_X>();
        {
            kunlun::DataCopySafeImpl(input_x, input_xGM[_offset_*INPUT_X_LW], _len_*INPUT_X_LW);
        }
         // Compute
        LocalTensor<TYPE_OUTPUT_Y> output_y = output_yQue.AllocTensor<TYPE_OUTPUT_Y>();
        {  
            if constexpr(std::is_same_v<TYPE_INPUT_X, int8_t>){
                kunlun::SyncMTE2MTE3();
                kunlun::DataCopySafeImpl(output_yGM[_offset_*OUTPUT_Y_LW], input_x, _len_*OUTPUT_Y_LW);
            } else if constexpr(std::is_same_v<TYPE_INPUT_X, uint8_t>){
                kunlun::SyncMTE2MTE3();
                kunlun::DataCopySafeImpl(output_yGM[_offset_*OUTPUT_Y_LW], input_x, _len_*OUTPUT_Y_LW);
            } else if constexpr(std::is_same_v<TYPE_INPUT_X, int32_t>){
                kunlun::SyncMTE2MTE3();
                kunlun::DataCopySafeImpl(output_yGM[_offset_*OUTPUT_Y_LW], input_x, _len_*OUTPUT_Y_LW);
            } else if constexpr(std::is_same_v<TYPE_INPUT_X, float>){
                auto x = input_x.template ReinterpretCast<float>();
                auto y = output_y.template ReinterpretCast<float>();
                auto y_int32 = output_y.template ReinterpretCast<int32_t>();
                kunlun::SyncMTE2S();
                AscendC::Cast(y_int32, x, AscendC::RoundMode::CAST_TRUNC, _len_);
                AscendC::Cast(y, y_int32, AscendC::RoundMode::CAST_NONE, _len_);
                Muls(y, y, (float)1, _len_);
                kunlun::SyncVMTE3();
                kunlun::DataCopySafeImpl(output_yGM[_offset_*OUTPUT_Y_LW], output_y, _len_*OUTPUT_Y_LW);
            } else if constexpr(std::is_same_v<TYPE_INPUT_X, half>){
                auto x = input_x.template ReinterpretCast<half>();
                auto y = output_y.template ReinterpretCast<half>();
                auto y_int16 = output_y.template ReinterpretCast<int16_t>();
                kunlun::SyncMTE2S();
                AscendC::Cast(y_int16, x, AscendC::RoundMode::CAST_TRUNC, _len_);
                AscendC::Cast(y, y_int16, AscendC::RoundMode::CAST_NONE, _len_);
                Muls(y, y, (half)1, _len_);
                kunlun::SyncVMTE3();
                kunlun::DataCopySafeImpl(output_yGM[_offset_*OUTPUT_Y_LW], output_y, _len_*OUTPUT_Y_LW);
            } else if constexpr(std::is_same_v<TYPE_INPUT_X, bfloat16_t>){
                auto x = input_x.template ReinterpretCast<bfloat16_t>();
                auto y = output_y.template ReinterpretCast<bfloat16_t>();
                kunlun::SyncMTE2S();
                auto y_int32 = output_y.template ReinterpretCast<int32_t>();
                auto y_float32 = output_y.template ReinterpretCast<float>();
                AscendC::Cast(y_int32, x, AscendC::RoundMode::CAST_TRUNC, _len_);
                AscendC::Cast(y_float32, y_int32, AscendC::RoundMode::CAST_NONE, _len_);
                AscendC::Cast(y, y_float32, AscendC::RoundMode::CAST_TRUNC, _len_);
                kunlun::SyncVMTE3();
                kunlun::DataCopySafeImpl(output_yGM[_offset_*OUTPUT_Y_LW], y, _len_*OUTPUT_Y_LW);
            }
        }
         // Free&Post Tensor
        input_xQue.FreeTensor(input_x);
        output_yQue.FreeTensor(output_y);
    }
private:
    GlobalTensor<TYPE_INPUT_X> input_xGM;
    GlobalTensor<TYPE_OUTPUT_Y> output_yGM;
    TQueBind<QuePosition::GM, QuePosition::VECIN, 2> input_xQue;
    TQueBind<QuePosition::VECOUT, QuePosition::GM, 2> output_yQue;
    uint32_t formerLen, formerNum, tailLen;
};