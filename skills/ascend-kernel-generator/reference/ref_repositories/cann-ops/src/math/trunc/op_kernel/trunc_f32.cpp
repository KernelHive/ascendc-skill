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
using namespace AscendC;
class KernelTruncF32 {
public:
    __aicore__ inline KernelTruncF32() {}
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
        input_xGM.SetGlobalBuffer((__gm__ float*)input_xAddr, totalLen);
        output_yGM.SetGlobalBuffer((__gm__ float*)output_yAddr, totalLen);
        
        pipe->InitBuffer(input_xQue, 2, formerLen*sizeof(float));
    }

    __aicore__ inline void Process()
    {
         // 读取变量
        const uint32_t formerNum32 = this->formerNum, formerLen32 = this->formerLen, tailLen32 = this->tailLen;
        {
            uint32_t t = 0;
            for (; t < formerNum32; ++t){  // 遍历大Tiling
                ProcessTrunc(
                    t*formerLen32,
                    formerLen32
                );
            }
            if(tailLen32) {  // 遍历小Tiling
                ProcessTrunc(
                    t*formerLen32,
                    tailLen32
                );
            }
        }
    }

private:
    __aicore__ inline void ProcessTrunc(const uint32_t& _offset_, const uint32_t& _len_){
         // CopyIn
        LocalTensor<float> input_x = input_xQue.AllocTensor<float>();
        {
            kunlun::DataCopySafeImpl(input_x, input_xGM[_offset_], _len_);
        }
        auto x = input_x.template ReinterpretCast<float>();
        auto x_i32 = input_x.template ReinterpretCast<int32_t>();
        kunlun::SyncMTE2S();
        AscendC::Cast(x_i32, x, AscendC::RoundMode::CAST_TRUNC, _len_);
        AscendC::Cast(x, x_i32, AscendC::RoundMode::CAST_NONE, _len_);
        kunlun::SyncVMTE3();
        kunlun::DataCopySafeImpl(output_yGM[_offset_], x, _len_);
        input_xQue.FreeTensor(input_x);
    }
private:
    GlobalTensor<float> input_xGM;
    GlobalTensor<float> output_yGM;
    TQueBind<QuePosition::GM, QuePosition::VECIN, 2> input_xQue;
    uint32_t formerLen, formerNum, tailLen;
};