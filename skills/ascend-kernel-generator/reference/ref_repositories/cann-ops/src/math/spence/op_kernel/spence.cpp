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
#include <type_traits>  

template<typename TYPE_X, typename TYPE_Y> 
class KernelSpence {  
    // Polynomial coefficients for Spence function approximation  
    static constexpr float A[] = {4.65128586073990045278E-5, 7.31589045238094711071E-3,   
                                 1.33847639578309018650E-1, 8.79691311754530315341E-1,  
                                 2.71149851196553469920E0, 4.25697156008121755724E0,  
                                 3.29771340985225106936E0, 1.00000000000000000126E0};  
                                 
    static constexpr float B[] = {6.90990488912553276999E-4, 2.54043763932544379113E-2,  
                                 2.82974860602568089943E-1, 1.41172597751831069617E0,  
                                 3.63800533345137075418E0, 5.03278880143316990390E0,  
                                 3.54771340985225096217E0, 9.99999999999999998740E-1};  
    // Buffer configuration  
    static constexpr int32_t BUFFER_NUM = 2;  
    static constexpr int32_t MUL_DE_NUM = 2; 
    static constexpr int32_t LOOP_NUM = 8; 
    static constexpr float PI_SQUARED_OVER_6 = 1.64493406684822643647f; // π²/6 (Spence function value at x=1)  
    static constexpr float THRESHOLD_ZERO = 0.0f;  
    static constexpr float THRESHOLD_HALF = 0.5f;  
    static constexpr float THRESHOLD_ONE = 1.0f;  
    static constexpr float THRESHOLD_ONE_HALF = 1.5f;  
    static constexpr float THRESHOLD_TWO = 2.0f;    
    static constexpr float SCALE_NEGATIVE = -1.0f;  
    static constexpr float SCALE_HALF_NEGATIVE = -0.5f;  
    static constexpr float SCALE_DUPLICATE = 2.0f;  
public:  
    __aicore__ inline KernelSpence() {}  
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain) {  
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");  
        this->blockLength = core_size + (AscendC::GetBlockNum() == AscendC::GetBlockIdx() + 1 ? core_remain : 0);  
        this->tileLength = block_size;  
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);  

        auto startPointer = core_size * AscendC::GetBlockIdx();  
        auto bufferlength = this->blockLength;  

        Gm_x.SetGlobalBuffer((__gm__ TYPE_X*)x + startPointer, bufferlength);  
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y + startPointer, bufferlength);  

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);  

        pipe.InitBuffer(Q_x, BUFFER_NUM, this->tileLength * sizeof(TYPE_X));  
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));  
        pipe.InitBuffer(B_w, this->tileLength * sizeof(float));  
        pipe.InitBuffer(B_tmp1, this->tileLength * sizeof(float));  
        pipe.InitBuffer(B_tmp2, this->tileLength * sizeof(float));  
        pipe.InitBuffer(B_tmp3, this->tileLength * sizeof(float));  
        pipe.InitBuffer(B_bits1, this->tileLength * sizeof(uint8_t));  
        pipe.InitBuffer(B_bits2, this->tileLength * sizeof(uint8_t));  
        pipe.InitBuffer(B_bits3, this->tileLength * sizeof(uint8_t));  
        if constexpr (!std::is_same_v<TYPE_X, float>) {  
            pipe.InitBuffer(B_fx, this->tileLength * sizeof(float));  
            pipe.InitBuffer(B_fy, this->tileLength * sizeof(float));  
        }  
    }  
    __aicore__ inline void Process() {  
        int32_t loopCount = this->tileNum;  
        for (int32_t i = 0; i < loopCount-1; i++) {  
            CopyIn(i, this->tileLength);  
            Compute(i, this->tileLength);  
            CopyOut(i, this->tileLength);  
        }  
        uint32_t length = this->blockLength - this->tileLength * (loopCount - 1);  
        CopyIn(loopCount - 1, length);  
        Compute(loopCount - 1, length);  
        CopyOut(loopCount - 1, length);  
    }  
private:  
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {  
        AscendC::LocalTensor<TYPE_X> x = Q_x.AllocTensor<TYPE_X>();  
        AscendC::DataCopy(x, Gm_x[progress * this->tileLength], length);  
        Q_x.EnQue(x);  
    }  
    __aicore__ inline void Compute(int32_t progress, uint32_t length) {  
        AscendC::LocalTensor<TYPE_X> x = Q_x.DeQue<TYPE_X>();  
        AscendC::LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();  
        if constexpr (std::is_same_v<TYPE_X, float>) {  
            Calculate(x, y, length);  
        }  
        else {  
            auto fx = B_fx.Get<float>();  
            auto fy = B_fy.Get<float>();  
            AscendC::Cast(fx, x, AscendC::RoundMode::CAST_NONE, length);  
            Calculate(fx, fy, length);  
            AscendC::Cast(y, fy, AscendC::RoundMode::CAST_NONE, length);  
        }  
        Q_x.FreeTensor(x);  
        Q_y.EnQue<TYPE_Y>(y);  
    }  
    __aicore__ inline void polevlf_A(AscendC::LocalTensor<float> &dst, AscendC::LocalTensor<float> &w, uint32_t length) {  
        AscendC::Duplicate(dst, THRESHOLD_ZERO, length);  
        for (int i = 0; i < LOOP_NUM; ++i) {  
            AscendC::Mul(dst, dst, w, length);  
            AscendC::Adds(dst, dst, A[i], length);  
        }  
    }  
    __aicore__ inline void polevlf_B(AscendC::LocalTensor<float> &dst, AscendC::LocalTensor<float> &w, uint32_t length) {  
        AscendC::Duplicate(dst, THRESHOLD_ZERO, length);  
        for (int i = 0; i < LOOP_NUM; ++i) {  
            AscendC::Mul(dst, dst, w, length);  
            AscendC::Adds(dst, dst, B[i], length);  
        }  
    }  
    __aicore__ inline void Calculate(AscendC::LocalTensor<float> &x, AscendC::LocalTensor<float> &y, uint32_t length) {  
        auto w = B_w.Get<float>();  
        auto tmp1 = B_tmp1.Get<float>(), tmp2 = B_tmp2.Get<float>(), tmp3 = B_tmp3.Get<float>();  
        auto bits1 = B_bits1.Get<uint8_t>(), bits2 = B_bits2.Get<uint8_t>(), bits3 = B_bits3.Get<uint8_t>();  

        // Check if x > 2  
        AscendC::Duplicate(tmp1, THRESHOLD_TWO, length);  
        AscendC::Compare(bits2, x, tmp1, AscendC::CMPMODE::GT, length);  
        AscendC::Duplicate(tmp2, THRESHOLD_ONE, length);  
        AscendC::Div(tmp3, tmp2, x, length);  
        AscendC::Select(tmp1, bits2, tmp3, THRESHOLD_ZERO, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, length);  
        AscendC::Not(bits3.ReinterpretCast<uint16_t>(), bits2.ReinterpretCast<uint16_t>(), length / MUL_DE_NUM);  
        AscendC::Select(tmp2, bits3, x, THRESHOLD_ZERO, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, length);  
        AscendC::Or(x.ReinterpretCast<uint16_t>(), tmp1.ReinterpretCast<uint16_t>(), tmp2.ReinterpretCast<uint16_t>(), length * MUL_DE_NUM);  

        // Check if x > 1.5  
        AscendC::Duplicate(tmp1, THRESHOLD_ONE_HALF, length);  
        AscendC::Compare(bits3, x, tmp1, AscendC::CMPMODE::GT, length);  
        AscendC::Adds(tmp3, tmp3, SCALE_NEGATIVE, length);  
        AscendC::Select(w, bits3, tmp3, THRESHOLD_ZERO, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, length);  
        AscendC::Or(bits2.ReinterpretCast<uint16_t>(), bits2.ReinterpretCast<uint16_t>(), bits3.ReinterpretCast<uint16_t>(), length / MUL_DE_NUM);  

        // Check if x < 0.5  
        AscendC::Duplicate(tmp1, THRESHOLD_HALF, length);  
        AscendC::Compare(bits1, x, tmp1, AscendC::CMPMODE::LT, length);  
        AscendC::Muls(tmp3, x, SCALE_NEGATIVE, length);  
        AscendC::Select(tmp1, bits1, tmp3, THRESHOLD_ZERO, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, length);  
        AscendC::Or(w.ReinterpretCast<uint16_t>(), w.ReinterpretCast<uint16_t>(), tmp1.ReinterpretCast<uint16_t>(), length * MUL_DE_NUM);  

        AscendC::Or(bits3.ReinterpretCast<uint16_t>(), bits1.ReinterpretCast<uint16_t>(), bits3.ReinterpretCast<uint16_t>(), length / MUL_DE_NUM);  
        AscendC::Not(bits3.ReinterpretCast<uint16_t>(), bits3.ReinterpretCast<uint16_t>(), length / MUL_DE_NUM);  
        AscendC::Adds(tmp1, x, SCALE_NEGATIVE, length);  
        AscendC::Select(tmp2, bits3, tmp1, THRESHOLD_ZERO, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, length);  
        AscendC::Or(w.ReinterpretCast<uint16_t>(), w.ReinterpretCast<uint16_t>(), tmp2.ReinterpretCast<uint16_t>(), length * MUL_DE_NUM);  

        // Polynomial evaluation for Spence function  
        polevlf_A(tmp1, w, length);  
        polevlf_B(tmp2, w, length);  
        AscendC::Muls(tmp1, tmp1, SCALE_NEGATIVE, length);  
        AscendC::Mul(tmp1, tmp1, w, length);  
        AscendC::Div(tmp2, tmp1, tmp2, length);  

        AscendC::Ln(tmp1, x, length);  
        AscendC::Adds(tmp3, tmp3, THRESHOLD_ONE, length);  
        AscendC::Ln(tmp3, tmp3, length);  
        AscendC::Mul(tmp1, tmp1, tmp3, length);  
        AscendC::Muls(tmp1, tmp1, SCALE_NEGATIVE, length);  
        AscendC::Adds(tmp1, tmp1, PI_SQUARED_OVER_6, length);  
        AscendC::Sub(tmp1, tmp1, tmp2, length);  
        AscendC::Select(tmp1, bits1, tmp1, THRESHOLD_ZERO, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, length);  
        AscendC::Not(bits1.ReinterpretCast<uint16_t>(), bits1.ReinterpretCast<uint16_t>(), length / MUL_DE_NUM);  
        AscendC::Select(tmp2, bits1, tmp2, THRESHOLD_ZERO, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, length);  
        AscendC::Or(tmp2.ReinterpretCast<uint16_t>(), tmp2.ReinterpretCast<uint16_t>(), tmp1.ReinterpretCast<uint16_t>(), length * MUL_DE_NUM);  

        AscendC::Ln(tmp1, x, length);  
        AscendC::Mul(tmp1, tmp1, tmp1, length);  
        AscendC::Muls(tmp1, tmp1, SCALE_HALF_NEGATIVE, length);  
        AscendC::Sub(tmp1, tmp1, tmp2, length);  
        AscendC::Select(tmp1, bits2, tmp1, THRESHOLD_ZERO, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, length);  
        AscendC::Not(bits2.ReinterpretCast<uint16_t>(), bits2.ReinterpretCast<uint16_t>(), length / MUL_DE_NUM);  
        AscendC::Select(tmp2, bits2, tmp2, THRESHOLD_ZERO, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, length);  
        AscendC::Or(y.ReinterpretCast<uint16_t>(), tmp1.ReinterpretCast<uint16_t>(), tmp2.ReinterpretCast<uint16_t>(), length * MUL_DE_NUM);  

        // Handle the special case when x = 0  
        AscendC::Duplicate(tmp1, THRESHOLD_ZERO, length);  
        AscendC::Compare(bits1, x, tmp1, AscendC::CMPMODE::NE, length);  
        AscendC::Select(y, bits1, y, PI_SQUARED_OVER_6, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, length);  
    }  
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {  
        AscendC::LocalTensor<TYPE_Y> y = Q_y.DeQue<TYPE_Y>();  
        AscendC::DataCopy(Gm_y[progress * this->tileLength], y, length);  
        Q_y.FreeTensor(y);  
    }  
private:  
    AscendC::TPipe pipe;  
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> Q_x;  
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> Q_y;  
    AscendC::TBuf<AscendC::QuePosition::VECCALC> B_w, B_fx, B_fy, B_tmp1, B_tmp2, B_tmp3;  
    AscendC::TBuf<AscendC::QuePosition::VECCALC> B_bits1, B_bits2, B_bits3;  
    AscendC::GlobalTensor<TYPE_X> Gm_x;  
    AscendC::GlobalTensor<TYPE_Y> Gm_y;  
    uint32_t blockLength;  
    uint32_t tileNum;  
    uint32_t tileLength;  
};  
namespace {
extern "C" __global__ __aicore__ void spence(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {  
    GET_TILING_DATA(tiling_data, tiling);  
    KernelSpence<DTYPE_X, DTYPE_Y> op;  
    op.Init(x, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);  
    op.Process();  
}  
}