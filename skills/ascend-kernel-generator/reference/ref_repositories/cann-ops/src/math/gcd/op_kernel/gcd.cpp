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
 * @file gcd.cpp
 */

#include "gcd_common.h"

template<typename T>
__aicore__ inline T MinVal(T a, T b) {
    return (a < b) ? a : b;
}

template<typename T>
__aicore__ inline T gcd_1(T a, T b) {
    a = (a < 0) ? -a : a;
    b = (b < 0) ? -b : b;

    if (a == 0) return b;
    if (b == 0) return a;

    while (b != 0) {
        T temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

template<typename T>
__aicore__ inline T gcd_2(T a, T b) {
    while (b != 0) {
        T temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}


using namespace AscendC;

template <typename T>
class KernelGcd
{
public:
    static constexpr int TILE_SIZE = GcdConfig<T>::TILE_SIZE;
    using FpT = typename GcdConfig<T>::FpType;
    static constexpr int TILE_SIZE_MASK = TILE_SIZE / 8;
    __aicore__ inline KernelGcd()
    {
    }
    __aicore__ inline void Init(int N0, int N1, int N2, int N3, int N4, int broadcast_mask, TPipe *pipe, GM_ADDR x1, GM_ADDR x2, GM_ADDR y)
    {
        N[0] = N0;
        N[1] = N1;
        N[2] = N2;
        N[3] = N3;
        N[4] = N4;
        for (int i = 0; i < 5;i++) {
            if (broadcast_mask & (1<<i)) {
                M[i] = 1;
            } else {
                M[i] = N[i];
            }
        }
        sizeX1 = 1;
        sizeX2 = 1;
        for (int i = 0;i < 5;i++) {
            sizeX1 *= N[i];
            sizeX2 *= M[i];
        }
        x1Gm.SetGlobalBuffer((__gm__ T *)x1, sizeX1 * sizeof(T));
        x2Gm.SetGlobalBuffer((__gm__ T *)x2, sizeX2 * sizeof(T));
        yGm.SetGlobalBuffer((__gm__ T *)y, sizeX1 * sizeof(T));

        if constexpr(std::is_same<T, int16_t>::value || std::is_same<T, int32_t>::value) {
            pipe->InitBuffer(tBufNext, 9 * TILE_SIZE * sizeof(T));
            pipe->InitBuffer(tBufMask, 10 * TILE_SIZE_MASK * sizeof(uint8_t));
            auto zeros = tBufNext.Get<T>()[5 * TILE_SIZE];
            Duplicate(zeros, (T) 0, TILE_SIZE);

            auto ones = tBufNext.Get<T>()[6 * TILE_SIZE];
            Duplicate(ones, (T) 1, TILE_SIZE);
        }

        pipe->InitBuffer(inX1, 1, TILE_SIZE * sizeof(T));
        pipe->InitBuffer(inX2, 1, TILE_SIZE * sizeof(T));
        pipe->InitBuffer(outY, 1, TILE_SIZE * sizeof(T));
    }

    __aicore__ inline void CopyInX1(int offset, int len) {
        LocalTensor<T> x1 = inX1.AllocTensor<T>();
        DataCopyExtParams copyParamsX;
        copyParamsX.blockCount = 1;
        copyParamsX.blockLen = len * sizeof(T);
        copyParamsX.srcStride = 0;
        copyParamsX.dstStride = 0;
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(x1, x1Gm[offset], copyParamsX, padParams);
        inX1.EnQue(x1);
    }

    __aicore__ inline void CopyInX2(int offset, int len) {
        LocalTensor<T> x2 = inX2.AllocTensor<T>();
        DataCopyExtParams copyParamsX;
        copyParamsX.blockCount = 1;
        copyParamsX.blockLen = len * sizeof(T);
        copyParamsX.srcStride = 0;
        copyParamsX.dstStride = 0;
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(x2, x2Gm[offset], copyParamsX, padParams);
        inX2.EnQue(x2);
    }

    __aicore__ inline void CopyOut(int offset, int len) {
        LocalTensor<T> y = outY.DeQue<T>();
        DataCopyExtParams copyParamsX;
        copyParamsX.blockCount = 1;
        copyParamsX.blockLen = len * sizeof(T);
        copyParamsX.srcStride = 0;
        copyParamsX.dstStride = 0;
        DataCopyPad(yGm[offset], y, copyParamsX);
        outY.FreeTensor(y);
    }

    template <typename U, int REP>
    __aicore__ inline void BinaryGcd(LocalTensor<U>& c, LocalTensor<U>& a, LocalTensor<U>& b, int len) {
        constexpr int32_t num_uint16 = sizeof(U) / sizeof(uint16_t);
    
        auto a0 = tBufNext.Get<U>();
        auto b0 = tBufNext.Get<U>()[1 * TILE_SIZE];
        auto a_div2 = tBufNext.Get<U>()[2 * TILE_SIZE];
        auto b_div2 = tBufNext.Get<U>()[3 * TILE_SIZE];
        auto ab_diff = tBufNext.Get<U>()[4 * TILE_SIZE];
        auto zeros = tBufNext.Get<U>()[5 * TILE_SIZE];
        auto ones = tBufNext.Get<U>()[6 * TILE_SIZE];
        auto c1 = tBufNext.Get<U>()[7 * TILE_SIZE];
        auto c2 = tBufNext.Get<U>()[8 * TILE_SIZE];
    
        auto even_a = tBufMask.Get<uint8_t>();
        auto even_b = tBufMask.Get<uint8_t>()[TILE_SIZE_MASK];
        auto odd_a = tBufMask.Get<uint8_t>()[2 * TILE_SIZE_MASK];
        auto odd_b = tBufMask.Get<uint8_t>()[3 * TILE_SIZE_MASK];
        auto nonzero_a = tBufMask.Get<uint8_t>()[4 * TILE_SIZE_MASK];
        auto zero_b = tBufMask.Get<uint8_t>()[5 * TILE_SIZE_MASK];
        auto even_ab = tBufMask.Get<uint8_t>()[6 * TILE_SIZE_MASK];
        auto odd_ab = tBufMask.Get<uint8_t>()[7 * TILE_SIZE_MASK];
        auto mask = tBufMask.Get<uint8_t>()[8 * TILE_SIZE_MASK];
        auto mask_not = tBufMask.Get<uint8_t>()[9 * TILE_SIZE_MASK];
    
        auto a_u16 = a.template ReinterpretCast<uint16_t>();
        auto b_u16 = b.template ReinterpretCast<uint16_t>();
        auto c_fp = c.template ReinterpretCast<FpT>();
        auto a_fp = a.template ReinterpretCast<FpT>();
        auto b_fp = b.template ReinterpretCast<FpT>();
    
        auto a0_u16 = a0.template ReinterpretCast<uint16_t>();
        auto b0_u16 = b0.template ReinterpretCast<uint16_t>();
        auto a0_fp = a0.template ReinterpretCast<FpT>();
        auto b0_fp = b0.template ReinterpretCast<FpT>();
    
        auto zeros_fp = zeros.template ReinterpretCast<FpT>();
        auto ones_fp = ones.template ReinterpretCast<FpT>();
        auto a_div2_fp = a_div2.template ReinterpretCast<FpT>();
        auto b_div2_fp = b_div2.template ReinterpretCast<FpT>();
        auto ab_diff_fp = ab_diff.template ReinterpretCast<FpT>();
        auto ones_u16 = ones.template ReinterpretCast<uint16_t>();
        
        auto c1_fp = c1.template ReinterpretCast<FpT>();
        auto c2_fp = c2.template ReinterpretCast<FpT>();
    
        auto even_a_u16 = even_a.template ReinterpretCast<uint16_t>();
        auto even_b_u16 = even_b.template ReinterpretCast<uint16_t>();
        auto odd_a_u16 = odd_a.template ReinterpretCast<uint16_t>();
        auto odd_b_u16 = odd_b.template ReinterpretCast<uint16_t>();
        auto nonzero_a_u16 = nonzero_a.template ReinterpretCast<uint16_t>();
        auto nonzero_a_fp = nonzero_a.template ReinterpretCast<FpT>();
        auto zero_b_u16 = zero_b.template ReinterpretCast<uint16_t>();
        auto even_ab_u16 = even_ab.template ReinterpretCast<uint16_t>();
        auto odd_ab_u16 = odd_ab.template ReinterpretCast<uint16_t>();
        auto mask_u16 = mask.template ReinterpretCast<uint16_t>();
        auto mask_not_u16 = mask_not.template ReinterpretCast<uint16_t>();
    
        Not(a0_u16, a_u16, num_uint16 * TILE_SIZE);
        Not(b0_u16, b_u16, num_uint16 * TILE_SIZE);
        Adds(a0, a0, (U)1, TILE_SIZE);
        Adds(b0, b0, (U)1, TILE_SIZE);
        Max(a, a, a0, TILE_SIZE);
        Max(b, b, b0, TILE_SIZE);
        
        Max(a0, a, b, TILE_SIZE);
        CompareScalar(nonzero_a, a0_fp, (FpT)0.0f, CMPMODE::NE, TILE_SIZE);
        Select(c_fp, nonzero_a, ones_fp, zeros_fp, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE_SIZE);
        ReduceMax(c1_fp, a0_fp, c2_fp, len, false);
        U max_val = c1.GetValue(0);

        for (int i = 0; i < REP && (max_val >> (i / 2)) > 0; i++) {
            Max(a0, a, b, TILE_SIZE);
            Min(b0, a, b, TILE_SIZE);

            And(a_u16, a0_u16, ones_u16, num_uint16 * TILE_SIZE);
            And(b_u16, b0_u16, ones_u16, num_uint16 * TILE_SIZE);
    
            ShiftRight(a_div2, a0, (U)1, TILE_SIZE);
            ShiftRight(b_div2, b0, (U)1, TILE_SIZE);
            Sub(ab_diff, a0, b0, TILE_SIZE);
            ShiftRight(ab_diff, ab_diff, (U)1, TILE_SIZE);
    
            CompareScalar(even_a, a_fp, (FpT)0.0f, CMPMODE::EQ, TILE_SIZE);
            CompareScalar(even_b, b_fp, (FpT)0.0f, CMPMODE::EQ, TILE_SIZE);
            CompareScalar(nonzero_a, a0_fp, (FpT)0.0f, CMPMODE::NE, TILE_SIZE);
            CompareScalar(zero_b, b0_fp, (FpT)0.0f, CMPMODE::EQ, TILE_SIZE);
            
            Mul(c1, c, a0, TILE_SIZE);
            Muls(c2, c, (U)2, TILE_SIZE);
    
            Not(odd_a_u16, even_a_u16, TILE_SIZE / 16);
            Not(odd_b_u16, even_b_u16, TILE_SIZE / 16);
            And(even_ab_u16, even_a_u16, even_b_u16, TILE_SIZE / 16);
            And(odd_ab_u16, odd_a_u16, odd_b_u16, TILE_SIZE / 16);
    
            And(mask_u16, nonzero_a_u16, zero_b_u16, TILE_SIZE / 16);

            // gcd(a, 0) = a
            Select(a_fp, mask, zeros_fp, a0_fp, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE_SIZE);
            Select(b_fp, mask, zeros_fp, b0_fp, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE_SIZE);
            Select(c_fp, mask, c1_fp, c_fp, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE_SIZE);
    
            Not(mask_u16, zero_b_u16, TILE_SIZE / 16);
            And(mask_not_u16, mask_u16, nonzero_a_u16, TILE_SIZE / 16);
            And(mask_u16, mask_not_u16, even_ab_u16, TILE_SIZE / 16);
    
            // gcd(2a, 2b) = 2 * gcd(a, b)
            Select(a_fp, mask, a_div2_fp, a_fp, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE_SIZE);
            Select(b_fp, mask, b_div2_fp, b_fp, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE_SIZE);
            Select(c_fp, mask, c2_fp, c_fp, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE_SIZE);
    
            Not(mask_u16, even_ab_u16, TILE_SIZE / 16);
            And(mask_not_u16, mask_u16, mask_not_u16, TILE_SIZE / 16);
            And(mask_u16, mask_not_u16, odd_ab_u16, TILE_SIZE / 16);
    
            // gcd(a, b) = gcd((a - b) / 2, b)
            Select(a_fp, mask, ab_diff_fp, a_fp, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE_SIZE);
    
            Not(mask_u16, odd_ab_u16, TILE_SIZE / 16);
            And(mask_not_u16, mask_u16, mask_not_u16, TILE_SIZE / 16);
            And(mask_u16, mask_not_u16, even_a_u16, TILE_SIZE / 16);
    
            // gcd(2a, b) = gcd(a, b)
            Select(a_fp, mask, a_div2_fp, a_fp, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE_SIZE);
    
            And(mask_u16, odd_a_u16, mask_not_u16, TILE_SIZE / 16);

             // gcd(a, 2b) = gcd(a, b)
            Select(b_fp, mask_u16, b_div2_fp, b_fp, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE_SIZE);
        }
    }

    __aicore__ inline void Compute(int len) {
        auto x1 = inX1.DeQue<T>();
        auto x2 = inX2.DeQue<T>();
        auto y = outY.AllocTensor<T>();
        if constexpr(std::is_same<T, int16_t>::value) {
            BinaryGcd<int16_t, 30>(y, x1, x2, len);
        } else if constexpr(std::is_same<T, int32_t>::value) {
            BinaryGcd<int32_t, 62>(y, x1, x2, len);
        } else {
            for (int i = 0;i < len;i++) {
                y.SetValue(i, gcd_1(x1.GetValue(i), x2.GetValue(i)));
            }
        }

        outY.EnQue(y);
        inX1.FreeTensor(x1);
        inX2.FreeTensor(x2);
    }

    __aicore__ inline void ProcessFast()
    {
        for (int i = GetBlockIdx() * TILE_SIZE;i < sizeX1;i+=TILE_SIZE * GetBlockNum()) {
            CopyInX1(i, MinVal(sizeX1 - i, TILE_SIZE));
            CopyInX2(i, MinVal(sizeX1 - i, TILE_SIZE));
            Compute(MinVal(sizeX1 - i, TILE_SIZE));
            CopyOut(i, MinVal(sizeX1 - i, TILE_SIZE));
        }
    }

    __aicore__ inline void ProcessSlow()
    {
        if (GetBlockIdx() != 0) {
            return;
        }
        // 预计算步长数组
        int y_stride[5], x2_stride[5];
        y_stride[4] = 1;        // 最内层维度步长为1
        x2_stride[4] = 1;

        for (int i = 3; i >= 0; i--) {
            y_stride[i] = y_stride[i+1] * N[i+1];
            x2_stride[i] = x2_stride[i+1] * M[i+1];
        }

        int i[5] = {0}; // i[0] to i[4]
        int y_idx = 0;
        int x2_idx = 0;

        for (int k = 0; k < sizeX1; k++) {
            T x1_val = x1Gm.GetValue(y_idx);
            T x2_val = x2Gm.GetValue(x2_idx);
            yGm.SetValue(y_idx, gcd_1(x1_val, x2_val));

            // 下标计算
            for (int d = 4; d >= 0; d--) {
                i[d]++;

                y_idx += y_stride[d];
                if (M[d] > 1) {
                    x2_idx += x2_stride[d];
                }

                if (i[d] < N[d]) {
                    break; 
                }
                
                i[d] = 0;
                y_idx -= N[d] * y_stride[d]; 
                if (M[d] > 1) {
                    x2_idx -= N[d] * x2_stride[d];
                }
            }
        }
    }

    __aicore__ inline void Process() {
        if (sizeX1 == sizeX2) {
            ProcessFast();
        } else {
            ProcessSlow();
        }
    }

    TQue<QuePosition::VECIN, 1> inX1, inX2;
    TQue<QuePosition::VECOUT, 1> outY;
    TBuf<TPosition::VECCALC> tBufNext, tBufMask;

    int N[5], M[5], sizeX1, sizeX2;
    GlobalTensor<T> x1Gm, x2Gm, yGm;
};


extern "C" __global__ __aicore__ void gcd(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelGcd<DTYPE_X1> op;
    TPipe pipe;
    op.Init(
        tiling_data.N0, tiling_data.N1, tiling_data.N2, tiling_data.N3, tiling_data.N4, tiling_data.broadcast_mask, &pipe,
        x1, x2, y
    );
    op.Process();
}