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
 * \file deep_norm.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "impl/dav_c220/kernel_operator_reg_others_impl.h"

using namespace AscendC;

static constexpr int32_t BUFFER_NUM = 1;
static constexpr uint32_t BLOCK_SIZE = 32;
static constexpr uint32_t FLOAT_BLOCK = 8;
static constexpr float ZERO = 0;
static constexpr uint32_t SINGLE_BLOCK = 1;
static constexpr uint32_t REPEAT_MAX = 64;

template <typename T>
class KernelDeepNorm {
public:
    __aicore__ inline KernelDeepNorm()
    {}
    __aicore__ inline uint32_t CEIL_DIV(uint32_t x, uint32_t y)
    {
        return y == 0 ? x : (x + y - 1) / y;
    }
    __aicore__ inline uint32_t ROUND_UP(uint32_t x)
    {
        return (x + blockNumEl - 1) / blockNumEl * blockNumEl;
    }
    __aicore__ inline uint32_t MIN(uint32_t x, uint32_t y)
    {
        return x < y ? x : y;
    }
    __aicore__ inline uint32_t MAX(uint32_t x, uint32_t y)
    {
        return x > y ? x : y;
    }

    __aicore__ inline float ReduceSumFP32(const LocalTensor<float> &src_local, int32_t count)
    {
        int32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(float);
        int32_t repeatTimes = count / elementNumPerRep;
        int32_t tailCount = count % elementNumPerRep;
        int32_t bodyCount = repeatTimes * elementNumPerRep;
#ifdef __CCE_KT_TEST__
        assert(count <= MAX_REPEAT_TIMES * elementNumPerRep);
#endif
        float value = 0.0;
        if (g_coreType == AIV) {
            if (likely(repeatTimes > 0)) {
                AscendCUtils::SetMask<float>(elementNumPerRep);
                ReduceSum(src_local, src_local, src_local, repeatTimes);
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                uint64_t acc_val = GetAccVal();
                value = *reinterpret_cast<float *>(&acc_val);
            }
            if (unlikely(tailCount != 0)) {
                AscendCUtils::SetMask<float>(tailCount);
                ReduceSum(src_local[bodyCount], src_local[bodyCount], src_local[bodyCount], 1);
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
#ifdef __CCE_KT_TEST__
                uint64_t acc_val = GetAccVal();
#else
                uint64_t acc_val = GetAccVal();
#endif
                value += *reinterpret_cast<float *>(&acc_val);
            }
        }
        return value;
    }

    __aicore__ inline void ReduceSumShort(const LocalTensor<float> &dst_local, const LocalTensor<float> &src_local,
        const LocalTensor<float> &tmp_local, int32_t len, int32_t repeat)
    {
        int32_t elementNum = BLOCK_SIZE / sizeof(float);
        int32_t maxRepeat = ONE_REPEAT_BYTE_SIZE / sizeof(float);
        int32_t tailCount = num_last_dim % elementNum;
        uint32_t index = 0;
        uint8_t repStride = len / FLOAT_BLOCK;

        int32_t repeatTimes = repeat / elementNum;
        int32_t bodyCount = repeatTimes * elementNum;
        int32_t repeatTail = repeat % elementNum * elementNum;

        Duplicate<float>(tmp_local, ZERO, repeat * elementNum);
        PipeBarrier<PIPE_V>();
        for (index = 0; index + elementNum <= num_last_dim; index += elementNum) {
            Add(tmp_local, tmp_local, src_local[index], elementNum, repeat, {1, 1, 1, 1, 1, repStride});
            PipeBarrier<PIPE_V>();
        }
        if (unlikely(tailCount != 0)) {
            Add(tmp_local, tmp_local, src_local[index], tailCount, repeat, {1, 1, 1, 1, 1, repStride});
        }
        PipeBarrier<PIPE_V>();
        if (repeatTimes != 0) {
            BlockReduceSum<float>(dst_local, tmp_local, repeatTimes, maxRepeat, 1, 1, elementNum);
        }
        if (repeatTail != 0) {
            BlockReduceSum<float>(
                dst_local[bodyCount], tmp_local[bodyCount * elementNum], 1, repeatTail, 1, 1, elementNum);
        }
    }

    __aicore__ inline float ReduceSumCustom(const LocalTensor<float> &src_local, int32_t count)
    {
#if __CCE_AICORE__ == 220
        return ReduceSumFP32(src_local, count);
#else
        LocalTensor<float> dst_local = x_buf_fp32.Get<float>();
        ReduceSum(dst_local, src_local, dst_local, count);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        float rstd_value = dst_local.GetValue(0);
        return rstd_value;
#endif
    }

    __aicore__ inline void InitBase(__gm__ uint8_t *x, __gm__ uint8_t *gx, __gm__ uint8_t *beta, __gm__ uint8_t *gamma,
        __gm__ uint8_t *mean, __gm__ uint8_t *rstd, __gm__ uint8_t *z, uint32_t num_core_, uint32_t num_Last_dim_,
        uint32_t num_first_dim_, uint32_t nl_first_dim_per_core_, uint32_t l_first_dim_per_core_,
        uint32_t first_dim_per_times_, uint32_t updated_last_dim_, uint32_t updated_last_times_, uint32_t eps_,
        uint32_t meanNum_, uint32_t alpha_)
    {
        //  store arguments
        databyte = sizeof(T);
        num_core = num_core_;
        num_last_dim = num_Last_dim_;
        num_first_dim = num_first_dim_;
        nl_first_dim_per_core = nl_first_dim_per_core_;
        l_first_dim_per_core = l_first_dim_per_core_;
        first_dim_per_times = first_dim_per_times_;
        updated_last_dim = updated_last_dim_;
        updated_last_times = updated_last_times_;
        blockNumEl = 32 / databyte;

        meanNum = *reinterpret_cast<float *>(&meanNum_);
        eps = *reinterpret_cast<float *>(&eps_);
        alphaVal = *reinterpret_cast<float *>(&alpha_);

        if (block_idx < num_core - 1) {
            row_work = nl_first_dim_per_core;
            row_step = first_dim_per_times;
        } else {
            row_work = l_first_dim_per_core;
            row_step = MIN(first_dim_per_times, row_work);
        }

        row_tail_ = (row_work % first_dim_per_times == 0) ? first_dim_per_times : (row_work % first_dim_per_times);

        uint32_t used_last_dim_num = (updated_last_dim == 0) ? num_last_dim : updated_last_dim;

        gm_offset_ = nl_first_dim_per_core * num_last_dim;
        gm_offset2_ = nl_first_dim_per_core;

        x_gm.SetGlobalBuffer((__gm__ T *)x + GetBlockIdx() * gm_offset_, row_work * num_last_dim);
        gx_gm.SetGlobalBuffer((__gm__ T *)gx + GetBlockIdx() * gm_offset_, row_work * num_last_dim);
        beta_gm.SetGlobalBuffer((__gm__ T *)beta, num_last_dim);
        gamma_gm.SetGlobalBuffer((__gm__ T *)gamma, num_last_dim);

        mean_gm.SetGlobalBuffer((__gm__ float *)mean + GetBlockIdx() * gm_offset2_, row_work);
        rstd_gm.SetGlobalBuffer((__gm__ float *)rstd + GetBlockIdx() * gm_offset2_, row_work);
        z_gm.SetGlobalBuffer((__gm__ T *)z + GetBlockIdx() * gm_offset_, row_work * num_last_dim);

        // input buffer
        pipe.InitBuffer(x_que, BUFFER_NUM, row_step * ROUND_UP(used_last_dim_num) * databyte);
        pipe.InitBuffer(gx_que, BUFFER_NUM, row_step * ROUND_UP(used_last_dim_num) * databyte);
        pipe.InitBuffer(beta_que, BUFFER_NUM, ROUND_UP(used_last_dim_num) * databyte);
        pipe.InitBuffer(gamma_que, BUFFER_NUM, ROUND_UP(used_last_dim_num) * databyte);

        // output buffer
        pipe.InitBuffer(mean_que_fp32, BUFFER_NUM, row_step * BLOCK_SIZE);
        pipe.InitBuffer(rstd_que_fp32, BUFFER_NUM, row_step * BLOCK_SIZE);
    }

    __aicore__ inline void InitShort(__gm__ uint8_t *x, __gm__ uint8_t *gx, __gm__ uint8_t *beta, __gm__ uint8_t *gamma,
        __gm__ uint8_t *mean, __gm__ uint8_t *rstd, __gm__ uint8_t *z, uint32_t num_core_, uint32_t num_Last_dim_,
        uint32_t num_first_dim_, uint32_t nl_first_dim_per_core_, uint32_t l_first_dim_per_core_,
        uint32_t first_dim_per_times_, uint32_t updated_last_dim_, uint32_t updated_last_times_, uint32_t eps_,
        uint32_t meanNum_, uint32_t alpha_)
    {
        InitBase(x,
            gx,
            beta,
            gamma,
            mean,
            rstd,
            z,
            num_core_,
            num_Last_dim_,
            num_first_dim_,
            nl_first_dim_per_core_,
            l_first_dim_per_core_,
            first_dim_per_times_,
            updated_last_dim_,
            updated_last_times_,
            eps_,
            meanNum_,
            alpha_);
        pipe.InitBuffer(z_que, BUFFER_NUM, row_step * ROUND_UP(num_last_dim) * databyte);
        pipe.InitBuffer(calc_buf_fp32, ROUND_UP(num_last_dim) * sizeof(float));
        // calc buffer
        if (databyte != 4) {
            pipe.InitBuffer(x_buf_fp32, row_step * sizeof(float) * ROUND_UP(num_last_dim));
            pipe.InitBuffer(y_buf_fp32, row_step * sizeof(float) * ROUND_UP(num_last_dim));
            pipe.InitBuffer(z_buf_fp32, row_step * sizeof(float) * ROUND_UP(num_last_dim));
        }
    }

    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *gx, __gm__ uint8_t *beta, __gm__ uint8_t *gamma,
        __gm__ uint8_t *mean, __gm__ uint8_t *rstd, __gm__ uint8_t *z, uint32_t num_core_, uint32_t num_Last_dim_,
        uint32_t num_first_dim_, uint32_t nl_first_dim_per_core_, uint32_t l_first_dim_per_core_,
        uint32_t first_dim_per_times_, uint32_t updated_last_dim_, uint32_t updated_last_times_, uint32_t eps_,
        uint32_t meanNum_, uint32_t alpha_)
    {
        InitBase(x,
            gx,
            beta,
            gamma,
            mean,
            rstd,
            z,
            num_core_,
            num_Last_dim_,
            num_first_dim_,
            nl_first_dim_per_core_,
            l_first_dim_per_core_,
            first_dim_per_times_,
            updated_last_dim_,
            updated_last_times_,
            eps_,
            meanNum_,
            alpha_);
        pipe.InitBuffer(z_que, BUFFER_NUM, row_step * ROUND_UP(num_last_dim) * databyte);
        pipe.InitBuffer(calc_buf_fp32, ROUND_UP(num_last_dim) * sizeof(float));
        // calc buffer
        pipe.InitBuffer(x_buf_fp32, sizeof(float) * ROUND_UP(num_last_dim));
        pipe.InitBuffer(y_buf_fp32, sizeof(float) * ROUND_UP(num_last_dim));
        pipe.InitBuffer(z_buf_fp32, sizeof(float) * ROUND_UP(num_last_dim));
        if (databyte != 4) {
            pipe.InitBuffer(calc_x_fp32, row_step * ROUND_UP(num_last_dim) * sizeof(float));
            pipe.InitBuffer(calc_y_fp32, row_step * ROUND_UP(num_last_dim) * sizeof(float));
        }
    }

    __aicore__ inline void InitExtra(__gm__ uint8_t *x, __gm__ uint8_t *gx, __gm__ uint8_t *beta, __gm__ uint8_t *gamma,
        __gm__ uint8_t *mean, __gm__ uint8_t *rstd, __gm__ uint8_t *z, uint32_t num_core_, uint32_t num_Last_dim_,
        uint32_t num_first_dim_, uint32_t nl_first_dim_per_core_, uint32_t l_first_dim_per_core_,
        uint32_t first_dim_per_times_, uint32_t updated_last_dim_, uint32_t updated_last_times_, uint32_t eps_,
        uint32_t meanNum_, uint32_t alpha_)
    {
        InitBase(x,
            gx,
            beta,
            gamma,
            mean,
            rstd,
            z,
            num_core_,
            num_Last_dim_,
            num_first_dim_,
            nl_first_dim_per_core_,
            l_first_dim_per_core_,
            first_dim_per_times_,
            updated_last_dim_,
            updated_last_times_,
            eps_,
            meanNum_,
            alpha_);
        // calc buffer
        pipe.InitBuffer(x_buf_fp32, sizeof(float) * ROUND_UP(updated_last_dim_));
        pipe.InitBuffer(y_buf_fp32, sizeof(float) * ROUND_UP(updated_last_dim_));
        pipe.InitBuffer(z_buf_fp32, sizeof(float) * ROUND_UP(updated_last_dim_));
        pipe.InitBuffer(z_que, BUFFER_NUM, row_step * ROUND_UP(num_last_dim) * databyte);
        pipe.InitBuffer(calc_buf_fp32, ROUND_UP(num_last_dim) * sizeof(float));
    }

    __aicore__ inline void InitCommon(__gm__ uint8_t *x, __gm__ uint8_t *gx, __gm__ uint8_t *beta,
        __gm__ uint8_t *gamma, __gm__ uint8_t *mean, __gm__ uint8_t *rstd, __gm__ uint8_t *z, uint32_t num_core_,
        uint32_t num_Last_dim_, uint32_t num_first_dim_, uint32_t nl_first_dim_per_core_,
        uint32_t l_first_dim_per_core_, uint32_t first_dim_per_times_, uint32_t updated_last_dim_,
        uint32_t updated_last_times_, uint32_t eps_, uint32_t meanNum_, uint32_t alpha_)
    {
        InitBase(x,
            gx,
            beta,
            gamma,
            mean,
            rstd,
            z,
            num_core_,
            num_Last_dim_,
            num_first_dim_,
            nl_first_dim_per_core_,
            l_first_dim_per_core_,
            first_dim_per_times_,
            updated_last_dim_,
            updated_last_times_,
            eps_,
            meanNum_,
            alpha_);
        // calc buffer
        pipe.InitBuffer(x_buf_fp32, sizeof(float) * ROUND_UP(updated_last_dim_));
        pipe.InitBuffer(y_buf_fp32, sizeof(float) * ROUND_UP(updated_last_dim_));
        pipe.InitBuffer(z_buf_fp32, sizeof(float) * ROUND_UP(updated_last_dim_));
        pipe.InitBuffer(z_que, BUFFER_NUM, row_step * ROUND_UP(updated_last_dim_) * databyte);
        pipe.InitBuffer(calc_buf_fp32, ROUND_UP(updated_last_dim_) * sizeof(float));
    }

    __aicore__ inline void ProcessFp16Short()
    {
        int32_t move_cnt = CEIL_DIV(row_work, row_step);
        int32_t loop = move_cnt - 1;
        CopyInBetaGamma(num_last_dim);
        LocalTensor<T> beta_local = beta_que.DeQue<T>();
        LocalTensor<T> gamma_local = gamma_que.DeQue<T>();
        for (int32_t i = 0; i < move_cnt; ++i) {
            if (i < loop) {
                CopyInXGX(i, row_step, num_last_dim);
                ComputeFp16Short(row_step, beta_local, gamma_local);
                CopyOut(i, row_step, num_last_dim);
            } else {
                CopyInXGX(i, row_tail_, num_last_dim);
                ComputeFp16Short(row_tail_, beta_local, gamma_local);
                CopyOut(i, row_tail_, num_last_dim);
            }
        }
        beta_que.FreeTensor(beta_local);
        gamma_que.FreeTensor(gamma_local);
    }

    __aicore__ inline void ProcessBf16Short()
    {
        int32_t move_cnt = CEIL_DIV(row_work, row_step);
        int32_t loop = move_cnt - 1;
        CopyInBetaGamma(num_last_dim);
        LocalTensor<T> beta_local = beta_que.DeQue<T>();
        LocalTensor<T> gamma_local = gamma_que.DeQue<T>();
        for (int32_t i = 0; i < move_cnt; ++i) {
            if (i < loop) {
                CopyInXGX(i, row_step, num_last_dim);
                ComputeBf16Short(row_step, beta_local, gamma_local);
                CopyOut(i, row_step, num_last_dim);
            } else {
                CopyInXGX(i, row_tail_, num_last_dim);
                ComputeBf16Short(row_tail_, beta_local, gamma_local);
                CopyOut(i, row_tail_, num_last_dim);
            }
        }
        beta_que.FreeTensor(beta_local);
        gamma_que.FreeTensor(gamma_local);
    }

    __aicore__ inline void ProcessFp32Short()
    {
        int32_t move_cnt = CEIL_DIV(row_work, row_step);
        int32_t loop = move_cnt - 1;
        CopyInBetaGamma(num_last_dim);
        LocalTensor<T> beta_local = beta_que.DeQue<T>();
        LocalTensor<T> gamma_local = gamma_que.DeQue<T>();
        for (int32_t i = 0; i < move_cnt; ++i) {
            if (i < loop) {
                CopyInXGX(i, row_step, num_last_dim);
                ComputeFp32Short(row_step, beta_local, gamma_local);
                CopyOut(i, row_step, num_last_dim);
            } else {
                CopyInXGX(i, row_tail_, num_last_dim);
                ComputeFp32Short(row_tail_, beta_local, gamma_local);
                CopyOut(i, row_tail_, num_last_dim);
            }
        }
        beta_que.FreeTensor(beta_local);
        gamma_que.FreeTensor(gamma_local);
    }

    __aicore__ inline void ProcessFp16LELimit()
    {
        int32_t move_cnt = CEIL_DIV(row_work, row_step);
        int32_t loop = move_cnt - 1;
        CopyInBetaGamma(num_last_dim);
        LocalTensor<T> beta_local = beta_que.DeQue<T>();
        LocalTensor<T> gamma_local = gamma_que.DeQue<T>();
        for (int32_t i = 0; i < move_cnt; ++i) {
            if (i < loop) {
                CopyInXGX(i, row_step, num_last_dim);
                ComputeFp16(row_step, beta_local, gamma_local);
                CopyOut(i, row_step, num_last_dim);
            } else {
                CopyInXGX(i, row_tail_, num_last_dim);
                ComputeFp16(row_tail_, beta_local, gamma_local);
                CopyOut(i, row_tail_, num_last_dim);
            }
        }
        beta_que.FreeTensor(beta_local);
        gamma_que.FreeTensor(gamma_local);
    }

    __aicore__ inline void ProcessBF16LELimit()
    {
        int32_t move_cnt = CEIL_DIV(row_work, row_step);
        int32_t loop = move_cnt - 1;
        CopyInBetaGamma(num_last_dim);
        LocalTensor<T> beta_local = beta_que.DeQue<T>();
        LocalTensor<T> gamma_local = gamma_que.DeQue<T>();
        for (int32_t i = 0; i < move_cnt; ++i) {
            if (i < loop) {
                CopyInXGX(i, row_step, num_last_dim);
                ComputeBf16(row_step, beta_local, gamma_local);
                CopyOut(i, row_step, num_last_dim);
            } else {
                CopyInXGX(i, row_tail_, num_last_dim);
                ComputeBf16(row_tail_, beta_local, gamma_local);
                CopyOut(i, row_tail_, num_last_dim);
            }
        }
        beta_que.FreeTensor(beta_local);
        gamma_que.FreeTensor(gamma_local);
    }

    __aicore__ inline void ProcessFp32LELimit()
    {
        int32_t move_cnt = CEIL_DIV(row_work, row_step);
        int32_t loop = move_cnt - 1;
        CopyInBetaGamma(num_last_dim);
        LocalTensor<T> beta_local = beta_que.DeQue<T>();
        LocalTensor<T> gamma_local = gamma_que.DeQue<T>();
        for (int32_t i = 0; i < move_cnt; ++i) {
            if (i < loop) {
                CopyInXGX(i, row_step, num_last_dim);
                ComputeFp32(row_step, beta_local, gamma_local);
                CopyOut(i, row_step, num_last_dim);
            } else {
                CopyInXGX(i, row_tail_, num_last_dim);
                ComputeFp32(row_tail_, beta_local, gamma_local);
                CopyOut(i, row_tail_, num_last_dim);
            }
        }
        beta_que.FreeTensor(beta_local);
        gamma_que.FreeTensor(gamma_local);
    }

    __aicore__ inline void ProcessFp16GTLimit()
    {
        int32_t move_cnt = CEIL_DIV(row_work, row_step);
        for (int32_t i = 0; i < move_cnt; ++i) {
            ExtraProcessFp16(i);
        }
    }

    __aicore__ inline void ProcessBf16GTLimit()
    {
        int32_t move_cnt = CEIL_DIV(row_work, row_step);
        for (int32_t i = 0; i < move_cnt; ++i) {
            ExtraProcessBf16(i);
        }
    }

    __aicore__ inline void ProcessFp32GTLimit()
    {
        int32_t move_cnt = CEIL_DIV(row_work, row_step);
        for (int32_t i = 0; i < move_cnt; ++i) {
            ExtraProcessFp32(i);
        }
    }

    __aicore__ inline void ProcessFp16Common()
    {
        int32_t move_cnt = CEIL_DIV(row_work, row_step);
        for (int32_t i = 0; i < move_cnt; ++i) {
            CommonProcessFp16(i);
        }
    }

    __aicore__ inline void ProcessBf16Common()
    {
        int32_t move_cnt = CEIL_DIV(row_work, row_step);
        for (int32_t i = 0; i < move_cnt; ++i) {
            CommonProcessBf16(i);
        }
    }

    __aicore__ inline void ProcessFp32Common()
    {
        int32_t move_cnt = CEIL_DIV(row_work, row_step);
        for (int32_t i = 0; i < move_cnt; ++i) {
            CommonProcessFp32(i);
        }
    }

private:
    // less or equal to limit case : fp16 / bf16 / fp32
    __aicore__ inline void CopyInXGX(int32_t proc_id, int32_t repeatTimes, int32_t length)
    {
        LocalTensor<T> x_local = x_que.AllocTensor<T>();
        LocalTensor<T> gx_local = gx_que.AllocTensor<T>();

        uint32_t offset = proc_id * row_step * num_last_dim;
#if __CCE_AICORE__ == 220
        DataCopyPadParams temp;
        DataCopyParams copyInput;
        DataCopyParams copyParams;

        temp.isPad = true;
        temp.paddingValue = 0;
        temp.rightPadding = ROUND_UP(length) - length;

        copyInput.blockCount = repeatTimes;
        copyInput.blockLen = length * sizeof(T);
        copyInput.srcStride = 0;
        copyInput.dstStride = 0;

        DataCopyPad(x_local, x_gm[offset], copyInput, temp);
        DataCopyPad(gx_local, gx_gm[offset], copyInput, temp);
#else
        for (uint32_t idx = 0; idx < repeatTimes; idx++) {
            DataCopy(x_local[idx * ROUND_UP(length)], x_gm[offset + idx * length], ROUND_UP(length));
            DataCopy(gx_local[idx * ROUND_UP(length)], gx_gm[offset + idx * length], ROUND_UP(length));
        }
#endif
        x_que.EnQue(x_local);
        gx_que.EnQue(gx_local);
    }

    __aicore__ inline void CopyInBetaGamma(int32_t length)
    {
        LocalTensor<T> beta_local = beta_que.AllocTensor<T>();
        LocalTensor<T> gamma_local = gamma_que.AllocTensor<T>();
#if __CCE_AICORE__ == 220
        DataCopyPadParams temp;
        DataCopyParams copyInput;
        DataCopyParams copyParams;

        copyParams.blockLen = length * sizeof(T);
        copyParams.blockCount = 1;
        DataCopyPad(beta_local, beta_gm, copyParams, temp);
        DataCopyPad(gamma_local, gamma_gm, copyParams, temp);
#else
        DataCopy(beta_local, beta_gm, ROUND_UP(length));
        DataCopy(gamma_local, gamma_gm, ROUND_UP(length));
#endif
        beta_que.EnQue(beta_local);
        gamma_que.EnQue(gamma_local);
    }

    __aicore__ inline void ComputeFp16(int32_t nums, LocalTensor<T> &beta_local, LocalTensor<T> &gamma_local)
    {
        // input
        LocalTensor<T> x_local = x_que.DeQue<T>();
        LocalTensor<T> gx_local = gx_que.DeQue<T>();
        LocalTensor<float> local_calc_fp32 = calc_buf_fp32.Get<float>();
        // output
        LocalTensor<T> z_local = z_que.AllocTensor<T>();
        LocalTensor<float> mean_local = mean_que_fp32.AllocTensor<float>();
        LocalTensor<float> rstd_local = rstd_que_fp32.AllocTensor<float>();

        // local temp
        LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();

        uint32_t realLen = ROUND_UP(num_last_dim);
        uint32_t stepSize = nums * realLen;

        LocalTensor<float> local_x_fp32 = calc_x_fp32.Get<float>();
        LocalTensor<float> local_y_fp32 = calc_y_fp32.Get<float>();

        Cast(local_y_fp32, x_local, RoundMode::CAST_NONE, stepSize);
        PipeBarrier<PIPE_V>();
        Cast(local_x_fp32, gx_local, RoundMode::CAST_NONE, stepSize);
        PipeBarrier<PIPE_V>();
        x_que.FreeTensor(x_local);
        gx_que.FreeTensor(gx_local);
        Axpy(local_x_fp32, local_y_fp32, alphaVal, stepSize);
        PipeBarrier<PIPE_V>();
        Muls(local_y_fp32, local_x_fp32, 1.0f, stepSize);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (int32_t rid = 0; rid < nums; ++rid) {
            uint32_t offset = rid * realLen;
            float mean_local_temp = ReduceSumCustom(local_y_fp32[offset], num_last_dim);
            event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(event_v_s);
            WaitFlag<HardEvent::V_S>(event_v_s);
            mean_local_temp = mean_local_temp * meanNum;
            mean_local[rid].SetValue(0, mean_local_temp);
            Adds(local_y_fp32[offset], local_x_fp32[offset], mean_local_temp * (-1), num_last_dim);
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        event_t event_s_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(event_s_mte3);
        WaitFlag<HardEvent::S_MTE3>(event_s_mte3);
        mean_que_fp32.EnQue(mean_local);

        Mul(local_x_fp32, local_y_fp32, local_y_fp32, stepSize);
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        Cast(z_local_fp32, gamma_local, RoundMode::CAST_NONE, num_last_dim);
        Cast(y_local_fp32, beta_local, RoundMode::CAST_NONE, num_last_dim);
        PipeBarrier<PIPE_V>();
        for (int32_t rid = 0; rid < nums; ++rid) {
            uint32_t offset = rid * realLen;

            float var_local_temp = ReduceSumCustom(local_x_fp32[offset], num_last_dim) * meanNum;
            float rstd_local_temp = 1 / sqrt(var_local_temp + eps);
            event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(event_v_s);
            WaitFlag<HardEvent::V_S>(event_v_s);
            rstd_local[rid].SetValue(0, rstd_local_temp);

            Muls(local_y_fp32[offset], local_y_fp32[offset], rstd_local_temp, num_last_dim);
            PipeBarrier<PIPE_V>();
            Mul(local_y_fp32[offset], local_y_fp32[offset], z_local_fp32, num_last_dim);
            PipeBarrier<PIPE_V>();
            Add(local_y_fp32[offset], local_y_fp32[offset], y_local_fp32, num_last_dim);
        }
        PipeBarrier<PIPE_V>();
        Cast(z_local, local_y_fp32, RoundMode::CAST_NONE, stepSize);
        SetFlag<HardEvent::S_MTE3>(event_s_mte3);
        WaitFlag<HardEvent::S_MTE3>(event_s_mte3);
        rstd_que_fp32.EnQue(rstd_local);
        z_que.EnQue(z_local);
    }

    __aicore__ inline void ComputeBf16(int32_t nums, LocalTensor<T> &beta_local, LocalTensor<T> &gamma_local)
    {
        // input
        LocalTensor<T> x_local = x_que.DeQue<T>();
        LocalTensor<T> gx_local = gx_que.DeQue<T>();
        LocalTensor<float> local_calc_fp32 = calc_buf_fp32.Get<float>();
        // output
        LocalTensor<T> z_local = z_que.AllocTensor<T>();
        LocalTensor<float> mean_local = mean_que_fp32.AllocTensor<float>();
        LocalTensor<float> rstd_local = rstd_que_fp32.AllocTensor<float>();

        // local temp
        LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();
        uint32_t realLen = ROUND_UP(num_last_dim);
        uint32_t stepSize = nums * realLen;

        LocalTensor<float> local_x_fp32 = calc_x_fp32.Get<float>();
        LocalTensor<float> local_y_fp32 = calc_y_fp32.Get<float>();

        Cast(local_y_fp32, x_local, RoundMode::CAST_NONE, stepSize);
        PipeBarrier<PIPE_V>();
        Cast(local_x_fp32, gx_local, RoundMode::CAST_NONE, stepSize);
        PipeBarrier<PIPE_V>();
        x_que.FreeTensor(x_local);
        gx_que.FreeTensor(gx_local);
        Axpy(local_x_fp32, local_y_fp32, alphaVal, stepSize);
        PipeBarrier<PIPE_V>();
        Muls(local_y_fp32, local_x_fp32, 1.0f, stepSize);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (int32_t rid = 0; rid < nums; ++rid) {
            uint32_t offset = rid * realLen;
            float mean_local_temp = ReduceSumCustom(local_y_fp32[offset], num_last_dim);
            event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(event_v_s);
            WaitFlag<HardEvent::V_S>(event_v_s);
            mean_local_temp = mean_local_temp * meanNum;
            mean_local[rid].SetValue(0, mean_local_temp);
            Adds(local_y_fp32[offset], local_x_fp32[offset], mean_local_temp * (-1), num_last_dim);
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        event_t event_s_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(event_s_mte3);
        WaitFlag<HardEvent::S_MTE3>(event_s_mte3);
        mean_que_fp32.EnQue(mean_local);
        Mul(local_x_fp32, local_y_fp32, local_y_fp32, stepSize);
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        Cast(z_local_fp32, gamma_local, RoundMode::CAST_NONE, num_last_dim);
        Cast(y_local_fp32, beta_local, RoundMode::CAST_NONE, num_last_dim);
        PipeBarrier<PIPE_V>();
        for (int32_t rid = 0; rid < nums; ++rid) {
            uint32_t offset = rid * realLen;
            float var_local_temp = ReduceSumCustom(local_x_fp32[offset], num_last_dim) * meanNum;
            float rstd_local_temp = 1 / sqrt(var_local_temp + eps);
            event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(event_v_s);
            WaitFlag<HardEvent::V_S>(event_v_s);
            rstd_local[rid].SetValue(0, rstd_local_temp);

            Muls(local_y_fp32[offset], local_y_fp32[offset], rstd_local_temp, num_last_dim);
            PipeBarrier<PIPE_V>();
            Mul(local_y_fp32[offset], local_y_fp32[offset], z_local_fp32, num_last_dim);
            PipeBarrier<PIPE_V>();
            Add(local_y_fp32[offset], local_y_fp32[offset], y_local_fp32, num_last_dim);
        }
        PipeBarrier<PIPE_V>();
        Cast(z_local, local_y_fp32, RoundMode::CAST_RINT, stepSize);
        SetFlag<HardEvent::S_MTE3>(event_s_mte3);
        WaitFlag<HardEvent::S_MTE3>(event_s_mte3);
        rstd_que_fp32.EnQue(rstd_local);
        z_que.EnQue(z_local);
    }

    __aicore__ inline void ComputeFp32(int32_t nums, LocalTensor<T> &beta_local, LocalTensor<T> &gamma_local)
    {
        // input
        LocalTensor<T> x_local = x_que.DeQue<T>();
        LocalTensor<T> gx_local = gx_que.DeQue<T>();
        LocalTensor<float> local_calc_fp32 = calc_buf_fp32.Get<float>();
        // output
        LocalTensor<T> z_local = z_que.AllocTensor<T>();
        LocalTensor<float> mean_local = mean_que_fp32.AllocTensor<float>();
        LocalTensor<float> rstd_local = rstd_que_fp32.AllocTensor<float>();

        // local temp
        LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();

        uint32_t realLen = ROUND_UP(num_last_dim);
        uint32_t stepSize = nums * realLen;

        Axpy(gx_local, x_local, alphaVal, stepSize);
        PipeBarrier<PIPE_V>();
        Muls(z_local, gx_local, 1.0f, stepSize);
        x_que.FreeTensor(x_local);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (int32_t rid = 0; rid < nums; ++rid) {
            uint32_t offset = rid * realLen;
            float mean_local_temp = ReduceSumCustom(z_local[offset], num_last_dim);
            event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(event_v_s);
            WaitFlag<HardEvent::V_S>(event_v_s);
            mean_local_temp = mean_local_temp * meanNum;
            mean_local[rid].SetValue(0, mean_local_temp);
            Adds(z_local[offset], gx_local[offset], mean_local_temp * (-1), num_last_dim);
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        event_t event_s_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(event_s_mte3);
        WaitFlag<HardEvent::S_MTE3>(event_s_mte3);
        mean_que_fp32.EnQue(mean_local);
        Mul(gx_local, z_local, z_local, stepSize);
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (int32_t rid = 0; rid < nums; ++rid) {
            uint32_t offset = rid * realLen;

            float var_local_temp = ReduceSumCustom(gx_local[offset], num_last_dim) * meanNum;
            float rstd_local_temp = 1 / sqrt(var_local_temp + eps);
            event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(event_v_s);
            WaitFlag<HardEvent::V_S>(event_v_s);
            rstd_local[rid].SetValue(0, rstd_local_temp);

            Muls(z_local[offset], z_local[offset], rstd_local_temp, num_last_dim);
            PipeBarrier<PIPE_V>();
            Mul(z_local[offset], z_local[offset], gamma_local, num_last_dim);
            PipeBarrier<PIPE_V>();
            Add(z_local[offset], z_local[offset], beta_local, num_last_dim);
        }
        PipeBarrier<PIPE_V>();
        gx_que.FreeTensor(gx_local);
        SetFlag<HardEvent::S_MTE3>(event_s_mte3);
        WaitFlag<HardEvent::S_MTE3>(event_s_mte3);
        rstd_que_fp32.EnQue(rstd_local);
        z_que.EnQue(z_local);
    }

    __aicore__ inline void ComputeFp16Short(int32_t nums, LocalTensor<T> &beta_local, LocalTensor<T> &gamma_local)
    {
        // input
        LocalTensor<T> x_local = x_que.DeQue<T>();
        LocalTensor<T> gx_local = gx_que.DeQue<T>();
        // output
        LocalTensor<T> z_local = z_que.AllocTensor<T>();
        LocalTensor<float> mean_local = mean_que_fp32.AllocTensor<float>();
        LocalTensor<float> rstd_local = rstd_que_fp32.AllocTensor<float>();

        // local temp
        LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();
        uint32_t realLen = ROUND_UP(num_last_dim);
        uint32_t stepSize = nums * realLen;
        Cast(x_local_fp32, x_local, RoundMode::CAST_NONE, stepSize);
        PipeBarrier<PIPE_V>();
        Cast(y_local_fp32, gx_local, RoundMode::CAST_NONE, stepSize);
        PipeBarrier<PIPE_V>();
        x_que.FreeTensor(x_local);
        gx_que.FreeTensor(gx_local);

        PrecisionComputeMeanShort(nums, z_local_fp32, x_local_fp32, y_local_fp32, mean_local);
        PipeBarrier<PIPE_V>();
        mean_que_fp32.EnQue(mean_local);
        PrecisionComputeRstdShort(nums, z_local_fp32, x_local_fp32, y_local_fp32, rstd_local);
        PipeBarrier<PIPE_V>();
        rstd_que_fp32.EnQue(rstd_local);

        Cast(x_local_fp32, gamma_local, RoundMode::CAST_NONE, num_last_dim);
        Cast(y_local_fp32, beta_local, RoundMode::CAST_NONE, num_last_dim);
        PrecisionComputeResultShort(nums, z_local_fp32, y_local_fp32, x_local_fp32);
        PipeBarrier<PIPE_V>();
        Cast(z_local, z_local_fp32, RoundMode::CAST_NONE, stepSize);
        z_que.EnQue(z_local);
    }

    __aicore__ inline void ComputeBf16Short(int32_t nums, LocalTensor<T> &beta_local, LocalTensor<T> &gamma_local)
    {
        // input
        LocalTensor<T> x_local = x_que.DeQue<T>();
        LocalTensor<T> gx_local = gx_que.DeQue<T>();
        // output
        LocalTensor<T> z_local = z_que.AllocTensor<T>();
        LocalTensor<float> mean_local = mean_que_fp32.AllocTensor<float>();
        LocalTensor<float> rstd_local = rstd_que_fp32.AllocTensor<float>();

        // local temp
        LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();
        uint32_t realLen = ROUND_UP(num_last_dim);
        uint32_t stepSize = nums * realLen;
        Cast(x_local_fp32, x_local, RoundMode::CAST_NONE, stepSize);
        PipeBarrier<PIPE_V>();
        Cast(y_local_fp32, gx_local, RoundMode::CAST_NONE, stepSize);
        PipeBarrier<PIPE_V>();
        x_que.FreeTensor(x_local);
        gx_que.FreeTensor(gx_local);

        PrecisionComputeMeanShort(nums, z_local_fp32, x_local_fp32, y_local_fp32, mean_local);
        PipeBarrier<PIPE_V>();
        mean_que_fp32.EnQue(mean_local);
        PrecisionComputeRstdShort(nums, z_local_fp32, x_local_fp32, y_local_fp32, rstd_local);
        PipeBarrier<PIPE_V>();
        rstd_que_fp32.EnQue(rstd_local);

        Cast(x_local_fp32, gamma_local, RoundMode::CAST_NONE, num_last_dim);
        Cast(y_local_fp32, beta_local, RoundMode::CAST_NONE, num_last_dim);
        PrecisionComputeResultShort(nums, z_local_fp32, y_local_fp32, x_local_fp32);
        PipeBarrier<PIPE_V>();
        Cast(z_local, z_local_fp32, RoundMode::CAST_RINT, stepSize);
        z_que.EnQue(z_local);
    }

    __aicore__ inline void ComputeFp32Short(int32_t nums, LocalTensor<T> &beta_local, LocalTensor<T> &gamma_local)
    {
        // input
        LocalTensor<T> x_local = x_que.DeQue<T>();
        LocalTensor<T> gx_local = gx_que.DeQue<T>();
        // output
        LocalTensor<T> z_local = z_que.AllocTensor<T>();
        LocalTensor<float> mean_local = mean_que_fp32.AllocTensor<float>();
        LocalTensor<float> rstd_local = rstd_que_fp32.AllocTensor<float>();

        PrecisionComputeMeanShort(nums, z_local, x_local, gx_local, mean_local);
        PipeBarrier<PIPE_V>();
        mean_que_fp32.EnQue(mean_local);
        PrecisionComputeRstdShort(nums, z_local, x_local, gx_local, rstd_local);
        PipeBarrier<PIPE_V>();
        rstd_que_fp32.EnQue(rstd_local);
        x_que.FreeTensor(x_local);
        gx_que.FreeTensor(gx_local);
        PrecisionComputeResultShort(nums, z_local, beta_local, gamma_local);
        z_que.EnQue(z_local);
    }

    __aicore__ inline void PrecisionComputeMeanShort(int32_t nums, LocalTensor<float> &z_local,
        LocalTensor<float> &x_local, LocalTensor<float> &gx_local, LocalTensor<float> &mean_local)
    {
        uint32_t realLen = ROUND_UP(num_last_dim);
        uint32_t stepSize = nums * realLen;

        // compute mean
        Axpy(gx_local, x_local, alphaVal, stepSize);
        PipeBarrier<PIPE_V>();
        Muls(z_local, gx_local, 1.0f, stepSize);
        PipeBarrier<PIPE_V>();
        ReduceSumShort(mean_local, z_local, x_local, realLen, nums);
        PipeBarrier<PIPE_V>();
        Muls(mean_local, mean_local, meanNum, nums);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (int32_t idx = nums - 1; idx >= 0; idx--) {
            uint32_t offset = idx * realLen;
            float meanTmp = mean_local.GetValue(idx);
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            Adds(z_local[offset], gx_local[offset], meanTmp * (-1), num_last_dim);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PrecisionComputeRstdShort(int32_t nums, LocalTensor<float> &z_local,
        LocalTensor<float> &x_local, LocalTensor<float> &gx_local, LocalTensor<float> &rstd_local)
    {
        uint32_t realLen = ROUND_UP(num_last_dim);
        uint32_t stepSize = nums * realLen;
        uint8_t repeatStride = realLen / FLOAT_BLOCK;

        uint32_t meanIter = FLOAT_BLOCK;
        uint32_t meanTail = num_last_dim % meanIter;
        uint32_t index = 0;
        uint32_t roundNums = (nums + FLOAT_BLOCK - 1) / FLOAT_BLOCK;

        // compute rstd
        Mul(gx_local, z_local, z_local, stepSize);
        PipeBarrier<PIPE_V>();
        ReduceSumShort(rstd_local, gx_local, x_local, realLen, nums);
        PipeBarrier<PIPE_V>();
        Muls(rstd_local, rstd_local, meanNum, nums);
        PipeBarrier<PIPE_V>();
        Adds(rstd_local, rstd_local, eps, nums);
        PipeBarrier<PIPE_V>();
        Sqrt(rstd_local, rstd_local, nums);
        Duplicate<float>(x_local, (float)1, nums);
        PipeBarrier<PIPE_V>();
        Div(rstd_local, x_local, rstd_local, nums);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (int32_t idx = nums - 1; idx >= 0; idx--) {
            uint32_t offset = idx * realLen;
            float rstdTmp = rstd_local.GetValue(idx);
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            Muls(z_local[offset], z_local[offset], rstdTmp, num_last_dim);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PrecisionComputeResultShort(
        int32_t nums, LocalTensor<float> &z_local, LocalTensor<float> &beta_local, LocalTensor<float> &gamma_local)
    {
        uint32_t realLen = ROUND_UP(num_last_dim);
        uint8_t repeatStride = realLen / FLOAT_BLOCK;
        uint32_t betaIter = REPEAT_MAX;
        uint32_t betaTail = num_last_dim % REPEAT_MAX;
        uint32_t index = 0;

        // compute result
        PipeBarrier<PIPE_V>();
        for (; index + betaIter <= num_last_dim; index += betaIter) {
            Mul(z_local[index],
                z_local[index],
                gamma_local[index],
                betaIter,
                nums,
                {1, 1, 1, repeatStride, repeatStride, 0});
            PipeBarrier<PIPE_V>();
            Add(z_local[index],
                z_local[index],
                beta_local[index],
                betaIter,
                nums,
                {1, 1, 1, repeatStride, repeatStride, 0});
        }
        if (betaTail != 0) {
            Mul(z_local[index],
                z_local[index],
                gamma_local[index],
                betaTail,
                nums,
                {1, 1, 1, repeatStride, repeatStride, 0});
            PipeBarrier<PIPE_V>();
            Add(z_local[index],
                z_local[index],
                beta_local[index],
                betaTail,
                nums,
                {1, 1, 1, repeatStride, repeatStride, 0});
        }
    }

    __aicore__ inline void CopyOut(int32_t proc_id, int32_t repeatTimes, int32_t length)
    {
        LocalTensor<T> z = z_que.DeQue<T>();
        LocalTensor<float> mean = mean_que_fp32.DeQue<float>();
        LocalTensor<float> rstd = rstd_que_fp32.DeQue<float>();

        uint32_t offset = proc_id * row_step * num_last_dim;
#if __CCE_AICORE__ == 220
        uint32_t offset2 = proc_id * row_step;

        DataCopyParams copyOutput;
        DataCopyParams copyParams;
        copyParams.blockLen = repeatTimes * sizeof(float);
        copyParams.blockCount = 1;
        DataCopyPad(mean_gm[offset2], mean, copyParams);
        DataCopyPad(rstd_gm[offset2], rstd, copyParams);

        copyOutput.blockCount = repeatTimes;
        copyOutput.blockLen = length * sizeof(T);

        copyOutput.srcStride = 0;
        copyOutput.dstStride = 0;
        DataCopyPad(z_gm[offset], z, copyOutput);
#else
        uint32_t blockNum = length / blockNumEl;
        uint32_t tail = length % blockNumEl;
        uint32_t blkLength = blockNum * blockNumEl;
        for (uint32_t idx = 0; idx < repeatTimes; idx++) {
            uint32_t curOffset = offset + idx * length;
            if (blockNum == 0) {
                break;
            }
            DataCopy(z_gm[curOffset], z[idx * ROUND_UP(length)], blkLength);
            if (tail != 0) {
                SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
                for (uint32_t i = 0; i < blockNumEl; i++) {
                    T tensorValue = z.GetValue(idx * ROUND_UP(length) + length - blockNumEl + i);
                    z.SetValue(idx * ROUND_UP(length) + i, tensorValue);
                }
                SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
                DataCopy(z_gm[curOffset + length - blockNumEl], z[idx * ROUND_UP(length)], blockNumEl);
            }
        }
#endif

        z_que.FreeTensor(z);
        mean_que_fp32.FreeTensor(mean);
        rstd_que_fp32.FreeTensor(rstd);
    }

    // larger than limit case : fp16 / bf16 / fp32
    __aicore__ inline void ExtraCopyXGX(uint32_t offset, int32_t size)
    {
        LocalTensor<T> x_local = x_que.AllocTensor<T>();
        LocalTensor<T> gx_local = gx_que.AllocTensor<T>();
#if __CCE_AICORE__ == 220
        DataCopyPadParams temp;
        DataCopyParams copyInput;

        copyInput.blockLen = size * sizeof(T);
        copyInput.blockCount = 1;

        DataCopyPad(x_local, x_gm[offset], copyInput, temp);
        DataCopyPad(gx_local, gx_gm[offset], copyInput, temp);
#else
        DataCopy(x_local, x_gm[offset], ROUND_UP(size));
        DataCopy(gx_local, gx_gm[offset], ROUND_UP(size));
#endif
        x_que.EnQue(x_local);
        gx_que.EnQue(gx_local);
    }

    __aicore__ inline void ExtraCopyBetaGamma(uint32_t offset, int32_t size)
    {
        LocalTensor<T> beta_local = beta_que.AllocTensor<T>();
        LocalTensor<T> gamma_local = gamma_que.AllocTensor<T>();
#if __CCE_AICORE__ == 220
        DataCopyPadParams temp;
        DataCopyParams copyParams;

        copyParams.blockLen = size * sizeof(T);
        copyParams.blockCount = 1;

        DataCopyPad(beta_local, beta_gm[offset], copyParams, temp);
        DataCopyPad(gamma_local, gamma_gm[offset], copyParams, temp);
#else
        DataCopy(beta_local, beta_gm[offset], ROUND_UP(size));
        DataCopy(gamma_local, gamma_gm[offset], ROUND_UP(size));
#endif
        beta_que.EnQue(beta_local);
        gamma_que.EnQue(gamma_local);
    }

    __aicore__ inline void ExtraProcessFp16(int32_t iter)
    {
        uint32_t offset = iter * row_step * num_last_dim;
        uint32_t lsize = num_last_dim - (updated_last_times - 1) * updated_last_dim;
        LocalTensor<float> sum_local;
        // Get Mean
        meanVal = 0;

        for (int i = 0; i < updated_last_times; i++) {
            uint32_t size = (i == updated_last_times - 1) ? lsize : updated_last_dim;
            ExtraCopyXGX(offset + i * updated_last_dim, size);
            ComputeMeanFp16Bf16(i, size);
            sum_local = z_buf_fp32.Get<float>();
            meanVal += ReduceSumCustom(sum_local, size);
        }
        meanVal = meanVal * meanNum;

        // Get Var
        varVal = 0;
        for (int j = 0; j < updated_last_times; j++) {
            uint32_t size = (j == updated_last_times - 1) ? lsize : updated_last_dim;
            ComputeVar(j, size);
            sum_local = z_buf_fp32.Get<float>();
            varVal += ReduceSumCustom(sum_local, size);
        }
        varVal = varVal * meanNum;
        // Get rstd
        varVal = 1 / sqrt(varVal + eps);
        // Get result
        LocalTensor<float> mean = mean_que_fp32.AllocTensor<float>();
        LocalTensor<float> rstd = rstd_que_fp32.AllocTensor<float>();
        event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event_v_s);
        WaitFlag<HardEvent::V_S>(event_v_s);
        mean.SetValue(0, meanVal);
        rstd.SetValue(0, varVal);
        event_t event_s_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(event_s_mte3);
        WaitFlag<HardEvent::S_MTE3>(event_s_mte3);

        LocalTensor<T> z_local = z_que.AllocTensor<T>();
        for (int k = 0; k < updated_last_times; k++) {
            uint32_t size = (k == updated_last_times - 1) ? lsize : updated_last_dim;
            ExtraCopyBetaGamma(k * updated_last_dim, size);
            ComputeResFp16Bf16(k, size);
            LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();
            PipeBarrier<PIPE_V>();
            Cast(z_local[k * updated_last_dim], z_local_fp32, RoundMode::CAST_NONE, size);
            PipeBarrier<PIPE_V>();
        }

        z_que.EnQue(z_local);
        mean_que_fp32.EnQue(mean);
        rstd_que_fp32.EnQue(rstd);
        ExtraCopyOut(iter);
    }

    __aicore__ inline void ExtraProcessBf16(int32_t iter)
    {
        uint32_t offset = iter * row_step * num_last_dim;
        uint32_t lsize = num_last_dim - (updated_last_times - 1) * updated_last_dim;
        LocalTensor<float> sum_local;
        // Get Mean
        meanVal = 0;

        for (int i = 0; i < updated_last_times; i++) {
            uint32_t size = (i == updated_last_times - 1) ? lsize : updated_last_dim;
            ExtraCopyXGX(offset + i * updated_last_dim, size);
            ComputeMeanFp16Bf16(i, size);
            sum_local = z_buf_fp32.Get<float>();
            meanVal += ReduceSumCustom(sum_local, size);
        }
        meanVal = meanVal * meanNum;

        // Get Var
        varVal = 0;
        for (int j = 0; j < updated_last_times; j++) {
            uint32_t size = (j == updated_last_times - 1) ? lsize : updated_last_dim;
            ComputeVar(j, size);
            sum_local = z_buf_fp32.Get<float>();
            varVal += ReduceSumCustom(sum_local, size);
        }
        varVal = varVal * meanNum;
        // Get rstd
        varVal = 1 / sqrt(varVal + eps);
        // Get result
        LocalTensor<float> mean = mean_que_fp32.AllocTensor<float>();
        LocalTensor<float> rstd = rstd_que_fp32.AllocTensor<float>();

        event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event_v_s);
        WaitFlag<HardEvent::V_S>(event_v_s);
        mean.SetValue(0, meanVal);
        rstd.SetValue(0, varVal);
        event_t event_s_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(event_s_mte3);
        WaitFlag<HardEvent::S_MTE3>(event_s_mte3);

        LocalTensor<T> z_local = z_que.AllocTensor<T>();
        for (int k = 0; k < updated_last_times; k++) {
            uint32_t size = (k == updated_last_times - 1) ? lsize : updated_last_dim;
            ExtraCopyBetaGamma(k * updated_last_dim, size);
            ComputeResFp16Bf16(k, size);
            LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();
            PipeBarrier<PIPE_V>();
            Cast(z_local[k * updated_last_dim], z_local_fp32, RoundMode::CAST_RINT, size);
            PipeBarrier<PIPE_V>();
        }
        z_que.EnQue(z_local);
        mean_que_fp32.EnQue(mean);
        rstd_que_fp32.EnQue(rstd);
        ExtraCopyOut(iter);
    }

    __aicore__ inline void ExtraProcessFp32(int32_t iter)
    {
        uint32_t offset = iter * row_step * num_last_dim;
        uint32_t lsize = num_last_dim - (updated_last_times - 1) * updated_last_dim;
        LocalTensor<float> sum_local;
        // Get Mean
        meanVal = 0;

        for (int i = 0; i < updated_last_times; i++) {
            uint32_t size = (i == updated_last_times - 1) ? lsize : updated_last_dim;
            ExtraCopyXGX(offset + i * updated_last_dim, size);
            ComputeMeanFp32(i, size);
            sum_local = z_buf_fp32.Get<float>();
            meanVal += ReduceSumCustom(sum_local, size);
        }
        meanVal = meanVal * meanNum;

        // Get Var
        varVal = 0;
        for (int j = 0; j < updated_last_times; j++) {
            uint32_t size = (j == updated_last_times - 1) ? lsize : updated_last_dim;
            ComputeVar(j, size);
            sum_local = z_buf_fp32.Get<float>();
            varVal += ReduceSumCustom(sum_local, size);
        }
        varVal = varVal * meanNum;
        // Get rstd
        varVal = 1 / sqrt(varVal + eps);
        // Get result
        LocalTensor<float> mean = mean_que_fp32.AllocTensor<float>();
        LocalTensor<float> rstd = rstd_que_fp32.AllocTensor<float>();

        event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event_v_s);
        WaitFlag<HardEvent::V_S>(event_v_s);
        mean.SetValue(0, meanVal);
        rstd.SetValue(0, varVal);
        event_t event_s_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(event_s_mte3);
        WaitFlag<HardEvent::S_MTE3>(event_s_mte3);

        LocalTensor<T> z_local = z_que.AllocTensor<T>();
        for (int k = 0; k < updated_last_times; k++) {
            uint32_t size = (k == updated_last_times - 1) ? lsize : updated_last_dim;
            ExtraCopyBetaGamma(k * updated_last_dim, size);
            ComputeResFp32(k, size);
            LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();
            float mask = 0;
            PipeBarrier<PIPE_V>();
            Adds(z_local[k * updated_last_dim], z_local_fp32, mask, size);
            PipeBarrier<PIPE_V>();
        }
        z_que.EnQue(z_local);
        mean_que_fp32.EnQue(mean);
        rstd_que_fp32.EnQue(rstd);
        ExtraCopyOut(iter);
    }

    // process shape in common
    __aicore__ inline void CommonProcessFp16(int32_t iter)
    {
        uint32_t offset = iter * row_step * num_last_dim;
        uint32_t lsize = num_last_dim - (updated_last_times - 1) * updated_last_dim;
        LocalTensor<float> sum_local;
        // Get Mean
        meanVal = 0;

        for (int i = 0; i < updated_last_times; i++) {
            uint32_t size = (i == updated_last_times - 1) ? lsize : updated_last_dim;
            ExtraCopyXGX(offset + i * updated_last_dim, size);
            ComputeMeanFp16Bf16(0, size);
            sum_local = z_buf_fp32.Get<float>();
            meanVal += ReduceSumCustom(sum_local, size);
        }
        meanVal = meanVal * meanNum;

        // Get Var
        varVal = 0;
        for (int j = 0; j < updated_last_times; j++) {
            uint32_t size = (j == updated_last_times - 1) ? lsize : updated_last_dim;
            ExtraCopyXGX(offset + j * updated_last_dim, size);
            ComputeVarCommonFp16(0, size);
            sum_local = z_buf_fp32.Get<float>();
            varVal += ReduceSumCustom(sum_local, size);
        }
        varVal = varVal * meanNum;
        // Get rstd
        varVal = 1 / sqrt(varVal + eps);
        // Get result
        LocalTensor<float> mean = mean_que_fp32.AllocTensor<float>();
        LocalTensor<float> rstd = rstd_que_fp32.AllocTensor<float>();
        mean.SetValue(0, meanVal);
        rstd.SetValue(0, varVal);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        mean_que_fp32.EnQue(mean);
        rstd_que_fp32.EnQue(rstd);
        CommonCopyOutParam(iter);

        for (int k = 0; k < updated_last_times; k++) {
            uint32_t size = (k == updated_last_times - 1) ? lsize : updated_last_dim;
            ExtraCopyXGX(offset + k * updated_last_dim, size);
            ExtraCopyBetaGamma(k * updated_last_dim, size);
            ComputeResCommonFp16(0, size);
            CommonCopyOutRes(offset + k * updated_last_dim, size);
        }
    }

    __aicore__ inline void CommonProcessBf16(int32_t iter)
    {
        uint32_t offset = iter * row_step * num_last_dim;
        uint32_t lsize = num_last_dim - (updated_last_times - 1) * updated_last_dim;
        LocalTensor<float> sum_local;
        // Get Mean
        meanVal = 0;

        for (int i = 0; i < updated_last_times; i++) {
            uint32_t size = (i == updated_last_times - 1) ? lsize : updated_last_dim;
            ExtraCopyXGX(offset + i * updated_last_dim, size);
            ComputeMeanFp16Bf16(0, size);
            sum_local = z_buf_fp32.Get<float>();
            meanVal += ReduceSumCustom(sum_local, size);
        }
        meanVal = meanVal * meanNum;

        // Get Var
        varVal = 0;
        for (int j = 0; j < updated_last_times; j++) {
            uint32_t size = (j == updated_last_times - 1) ? lsize : updated_last_dim;
            ExtraCopyXGX(offset + j * updated_last_dim, size);
            ComputeVarCommonFp16(0, size);
            sum_local = z_buf_fp32.Get<float>();
            varVal += ReduceSumCustom(sum_local, size);
        }
        varVal = varVal * meanNum;
        // Get rstd
        varVal = 1 / sqrt(varVal + eps);
        // Get result
        LocalTensor<float> mean = mean_que_fp32.AllocTensor<float>();
        LocalTensor<float> rstd = rstd_que_fp32.AllocTensor<float>();
        mean.SetValue(0, meanVal);
        rstd.SetValue(0, varVal);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        mean_que_fp32.EnQue(mean);
        rstd_que_fp32.EnQue(rstd);
        CommonCopyOutParam(iter);

        for (int k = 0; k < updated_last_times; k++) {
            uint32_t size = (k == updated_last_times - 1) ? lsize : updated_last_dim;
            ExtraCopyXGX(offset + k * updated_last_dim, size);
            ExtraCopyBetaGamma(k * updated_last_dim, size);
            ComputeResCommonBf16(0, size);
            CommonCopyOutRes(offset + k * updated_last_dim, size);
        }
    }

    __aicore__ inline void CommonProcessFp32(int32_t iter)
    {
        uint32_t offset = iter * row_step * num_last_dim;
        uint32_t lsize = num_last_dim - (updated_last_times - 1) * updated_last_dim;
        LocalTensor<float> sum_local;
        // Get Mean
        meanVal = 0;

        for (int i = 0; i < updated_last_times; i++) {
            uint32_t size = (i == updated_last_times - 1) ? lsize : updated_last_dim;
            ExtraCopyXGX(offset + i * updated_last_dim, size);
            ComputeMeanFp32(0, size);
            sum_local = z_buf_fp32.Get<float>();
            meanVal += ReduceSumCustom(sum_local, size);
        }
        meanVal = meanVal * meanNum;

        // Get Var
        varVal = 0;
        for (int j = 0; j < updated_last_times; j++) {
            uint32_t size = (j == updated_last_times - 1) ? lsize : updated_last_dim;
            ExtraCopyXGX(offset + j * updated_last_dim, size);
            ComputeVarCommonFp32(0, size);
            sum_local = z_buf_fp32.Get<float>();
            varVal += ReduceSumCustom(sum_local, size);
        }
        varVal = varVal * meanNum;
        // Get rstd
        varVal = 1 / sqrt(varVal + eps);
        // Get result
        LocalTensor<float> mean = mean_que_fp32.AllocTensor<float>();
        LocalTensor<float> rstd = rstd_que_fp32.AllocTensor<float>();
        mean.SetValue(0, meanVal);
        rstd.SetValue(0, varVal);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        mean_que_fp32.EnQue(mean);
        rstd_que_fp32.EnQue(rstd);
        CommonCopyOutParam(iter);

        for (int k = 0; k < updated_last_times; k++) {
            uint32_t size = (k == updated_last_times - 1) ? lsize : updated_last_dim;
            ExtraCopyXGX(offset + k * updated_last_dim, size);
            ExtraCopyBetaGamma(k * updated_last_dim, size);
            ComputeResCommonFp32(0, size);
            CommonCopyOutRes(offset + k * updated_last_dim, size);
        }
    }

    // sub functions
    __aicore__ inline void ComputeMeanFp16Bf16(int32_t index, int32_t size)
    {
        LocalTensor<T> x_local = x_que.DeQue<T>();
        LocalTensor<T> gx_local = gx_que.DeQue<T>();
        LocalTensor<float> new_x_local_fp32 = calc_buf_fp32.Get<float>();
        LocalTensor<float> mean_local_fp32 = z_buf_fp32.Get<float>();

        LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();

        PipeBarrier<PIPE_V>();
        Cast(x_local_fp32, x_local, RoundMode::CAST_NONE, size);  // 16bit -> 32bit
        Cast(y_local_fp32, gx_local, RoundMode::CAST_NONE, size);
        PipeBarrier<PIPE_V>();
        Muls(x_local_fp32, x_local_fp32, alphaVal, size);
        x_que.FreeTensor(x_local);
        gx_que.FreeTensor(gx_local);

        // check x val
        PipeBarrier<PIPE_V>();
        Add(new_x_local_fp32[index * updated_last_dim], x_local_fp32, y_local_fp32, size);  // x_new = x_new + gx
        PipeBarrier<PIPE_V>();
        Muls(mean_local_fp32, new_x_local_fp32[index * updated_last_dim], 1.0f, size);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeMeanFp32(int32_t index, int32_t size)
    {
        LocalTensor<T> x_local = x_que.DeQue<T>();
        LocalTensor<T> gx_local = gx_que.DeQue<T>();
        LocalTensor<float> new_x_local_fp32 = calc_buf_fp32.Get<float>();
        LocalTensor<float> mean_local_fp32 = z_buf_fp32.Get<float>();

        LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();

        PipeBarrier<PIPE_V>();
        Muls(x_local_fp32, x_local, alphaVal, size);

        // check x val
        PipeBarrier<PIPE_V>();
        Add(new_x_local_fp32[index * updated_last_dim], x_local_fp32, gx_local, size);  // x_new = x_new + gx
        PipeBarrier<PIPE_V>();
        Muls(mean_local_fp32, new_x_local_fp32[index * updated_last_dim], 1.0f, size);
        PipeBarrier<PIPE_V>();

        x_que.FreeTensor(x_local);
        gx_que.FreeTensor(gx_local);
    }

    __aicore__ inline void ComputeVar(int32_t index, int32_t size)
    {
        LocalTensor<float> new_x_fp32 = calc_buf_fp32.Get<float>();

        LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();

        PipeBarrier<PIPE_V>();
        Adds(new_x_fp32[index * updated_last_dim], new_x_fp32[index * updated_last_dim], meanVal * (-1), size);
        PipeBarrier<PIPE_V>();
        Mul(z_local_fp32, new_x_fp32[index * updated_last_dim], new_x_fp32[index * updated_last_dim], size);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeVarCommonFp16(int32_t index, int32_t size)
    {
        LocalTensor<T> x_local = x_que.DeQue<T>();
        LocalTensor<T> gx_local = gx_que.DeQue<T>();

        LocalTensor<float> new_x_fp32 = calc_buf_fp32.Get<float>();
        LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();

        PipeBarrier<PIPE_V>();
        Cast(x_local_fp32, x_local, RoundMode::CAST_NONE, size);  // 16bit -> 32bit
        Cast(y_local_fp32, gx_local, RoundMode::CAST_NONE, size);
        PipeBarrier<PIPE_V>();
        x_que.FreeTensor(x_local);
        gx_que.FreeTensor(gx_local);
        Muls(x_local_fp32, x_local_fp32, alphaVal, size);

        // check x val
        PipeBarrier<PIPE_V>();
        Add(new_x_fp32[index * updated_last_dim], x_local_fp32, y_local_fp32, size);  // x_new = x_new + gx

        PipeBarrier<PIPE_V>();
        Adds(new_x_fp32[index * updated_last_dim], new_x_fp32[index * updated_last_dim], meanVal * (-1), size);
        PipeBarrier<PIPE_V>();
        Mul(z_local_fp32, new_x_fp32[index * updated_last_dim], new_x_fp32[index * updated_last_dim], size);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeVarCommonFp32(int32_t index, int32_t size)
    {
        LocalTensor<T> x_local = x_que.DeQue<T>();
        LocalTensor<T> gx_local = gx_que.DeQue<T>();
        LocalTensor<float> new_x_fp32 = calc_buf_fp32.Get<float>();

        LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();

        PipeBarrier<PIPE_V>();
        Muls(x_local_fp32, x_local, alphaVal, size);

        // check x val
        PipeBarrier<PIPE_V>();
        Add(new_x_fp32[index * updated_last_dim], x_local_fp32, gx_local, size);  // x_new = x_new + gx
        PipeBarrier<PIPE_V>();
        x_que.FreeTensor(x_local);
        gx_que.FreeTensor(gx_local);
        Adds(new_x_fp32[index * updated_last_dim], new_x_fp32[index * updated_last_dim], meanVal * (-1), size);
        PipeBarrier<PIPE_V>();
        Mul(z_local_fp32, new_x_fp32[index * updated_last_dim], new_x_fp32[index * updated_last_dim], size);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeResFp16Bf16(int32_t index, int32_t size)
    {
        LocalTensor<T> beta_local = beta_que.DeQue<T>();
        LocalTensor<T> gamma_local = gamma_que.DeQue<T>();

        // local temp
        LocalTensor<float> new_x_fp32 = calc_buf_fp32.Get<float>();
        LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();

        PipeBarrier<PIPE_V>();
        Muls(z_local_fp32, new_x_fp32[index * updated_last_dim], varVal, size);
        Cast(x_local_fp32, gamma_local, RoundMode::CAST_NONE, size);
        PipeBarrier<PIPE_V>();
        Mul(z_local_fp32, z_local_fp32, x_local_fp32, size);
        Cast(y_local_fp32, beta_local, RoundMode::CAST_NONE, size);
        PipeBarrier<PIPE_V>();
        Add(z_local_fp32, z_local_fp32, y_local_fp32, size);
        PipeBarrier<PIPE_V>();

        beta_que.FreeTensor(beta_local);
        gamma_que.FreeTensor(gamma_local);
    }

    __aicore__ inline void ComputeResFp32(int32_t index, int32_t size)
    {
        LocalTensor<T> beta_local = beta_que.DeQue<T>();
        LocalTensor<T> gamma_local = gamma_que.DeQue<T>();

        // local temp
        LocalTensor<float> new_x_fp32 = calc_buf_fp32.Get<float>();
        LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();

        PipeBarrier<PIPE_V>();
        Muls(z_local_fp32, new_x_fp32[index * updated_last_dim], varVal, size);
        PipeBarrier<PIPE_V>();
        Mul(z_local_fp32, z_local_fp32, gamma_local, size);
        PipeBarrier<PIPE_V>();
        Add(z_local_fp32, z_local_fp32, beta_local, size);
        PipeBarrier<PIPE_V>();

        beta_que.FreeTensor(beta_local);
        gamma_que.FreeTensor(gamma_local);
    }

    __aicore__ inline void ComputeResCommonFp16(int32_t index, int32_t size)
    {
        LocalTensor<T> x_local = x_que.DeQue<T>();
        LocalTensor<T> gx_local = gx_que.DeQue<T>();
        LocalTensor<T> beta_local = beta_que.DeQue<T>();
        LocalTensor<T> gamma_local = gamma_que.DeQue<T>();
        LocalTensor<T> z_local = z_que.AllocTensor<T>();
        // local temp
        LocalTensor<float> new_x_fp32 = calc_buf_fp32.Get<float>();
        LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();

        PipeBarrier<PIPE_V>();
        Cast(x_local_fp32, x_local, RoundMode::CAST_NONE, size);  // 16bit -> 32bit
        Cast(y_local_fp32, gx_local, RoundMode::CAST_NONE, size);
        PipeBarrier<PIPE_V>();
        x_que.FreeTensor(x_local);
        gx_que.FreeTensor(gx_local);
        Muls(x_local_fp32, x_local_fp32, alphaVal, size);

        // check x val
        PipeBarrier<PIPE_V>();
        Add(new_x_fp32[index * updated_last_dim], x_local_fp32, y_local_fp32, size);  // x_new = x_new + gx
        PipeBarrier<PIPE_V>();
        Adds(new_x_fp32[index * updated_last_dim], new_x_fp32[index * updated_last_dim], meanVal * (-1), size);

        PipeBarrier<PIPE_V>();
        Muls(y_local_fp32, new_x_fp32[index * updated_last_dim], varVal, size);
        PipeBarrier<PIPE_V>();
        Cast(x_local_fp32, gamma_local, RoundMode::CAST_NONE, size);
        PipeBarrier<PIPE_V>();
        Mul(y_local_fp32, x_local_fp32, y_local_fp32, size);
        PipeBarrier<PIPE_V>();
        Cast(x_local_fp32, beta_local, RoundMode::CAST_NONE, size);
        PipeBarrier<PIPE_V>();
        Add(z_local_fp32, y_local_fp32, x_local_fp32, size);
        PipeBarrier<PIPE_V>();
        Cast(z_local[index * updated_last_dim], z_local_fp32, RoundMode::CAST_NONE, size);

        beta_que.FreeTensor(beta_local);
        gamma_que.FreeTensor(gamma_local);
        z_que.EnQue(z_local);
    }

    __aicore__ inline void ComputeResCommonBf16(int32_t index, int32_t size)
    {
        LocalTensor<T> x_local = x_que.DeQue<T>();
        LocalTensor<T> gx_local = gx_que.DeQue<T>();
        LocalTensor<T> beta_local = beta_que.DeQue<T>();
        LocalTensor<T> gamma_local = gamma_que.DeQue<T>();
        LocalTensor<T> z_local = z_que.AllocTensor<T>();
        // local temp
        LocalTensor<float> new_x_fp32 = calc_buf_fp32.Get<float>();
        LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();

        PipeBarrier<PIPE_V>();
        Cast(x_local_fp32, x_local, RoundMode::CAST_NONE, size);  // 16bit -> 32bit
        Cast(y_local_fp32, gx_local, RoundMode::CAST_NONE, size);
        PipeBarrier<PIPE_V>();
        x_que.FreeTensor(x_local);
        gx_que.FreeTensor(gx_local);
        Muls(x_local_fp32, x_local_fp32, alphaVal, size);

        // check x val
        PipeBarrier<PIPE_V>();
        Add(new_x_fp32[index * updated_last_dim], x_local_fp32, y_local_fp32, size);  // x_new = x_new + gx
        PipeBarrier<PIPE_V>();
        Adds(new_x_fp32[index * updated_last_dim], new_x_fp32[index * updated_last_dim], meanVal * (-1), size);

        PipeBarrier<PIPE_V>();
        Muls(y_local_fp32, new_x_fp32[index * updated_last_dim], varVal, size);
        PipeBarrier<PIPE_V>();
        Cast(x_local_fp32, gamma_local, RoundMode::CAST_NONE, size);
        PipeBarrier<PIPE_V>();
        Mul(y_local_fp32, x_local_fp32, y_local_fp32, size);
        PipeBarrier<PIPE_V>();
        Cast(x_local_fp32, beta_local, RoundMode::CAST_NONE, size);
        PipeBarrier<PIPE_V>();
        Add(z_local_fp32, y_local_fp32, x_local_fp32, size);
        PipeBarrier<PIPE_V>();
        Cast(z_local[index * updated_last_dim], z_local_fp32, RoundMode::CAST_RINT, size);

        beta_que.FreeTensor(beta_local);
        gamma_que.FreeTensor(gamma_local);
        z_que.EnQue(z_local);
    }

    __aicore__ inline void ComputeResCommonFp32(int32_t index, int32_t size)
    {
        LocalTensor<T> x_local = x_que.DeQue<T>();
        LocalTensor<T> gx_local = gx_que.DeQue<T>();
        LocalTensor<T> beta_local = beta_que.DeQue<T>();
        LocalTensor<T> gamma_local = gamma_que.DeQue<T>();
        LocalTensor<T> z_local = z_que.AllocTensor<T>();
        // local temp
        LocalTensor<float> new_x_fp32 = calc_buf_fp32.Get<float>();
        LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();

        PipeBarrier<PIPE_V>();
        Muls(x_local_fp32, x_local, alphaVal, size);

        // check x val
        PipeBarrier<PIPE_V>();
        Add(new_x_fp32[index * updated_last_dim], x_local_fp32, gx_local, size);  // x_new = x_new + gx
        PipeBarrier<PIPE_V>();
        x_que.FreeTensor(x_local);
        gx_que.FreeTensor(gx_local);

        Adds(new_x_fp32[index * updated_last_dim], new_x_fp32[index * updated_last_dim], meanVal * (-1), size);
        PipeBarrier<PIPE_V>();
        Muls(y_local_fp32, new_x_fp32[index * updated_last_dim], varVal, size);
        PipeBarrier<PIPE_V>();
        Mul(y_local_fp32, y_local_fp32, gamma_local, size);
        PipeBarrier<PIPE_V>();
        Add(z_local, y_local_fp32, beta_local, size);
        PipeBarrier<PIPE_V>();

        beta_que.FreeTensor(beta_local);
        gamma_que.FreeTensor(gamma_local);
        z_que.EnQue(z_local);
    }

    __aicore__ inline void ExtraCopyOut(int32_t offset)
    {
        CommonCopyOutRes(offset * num_last_dim, num_last_dim);
        CommonCopyOutParam(offset);
    }

    __aicore__ inline void CommonCopyOutRes(int32_t offset, int32_t length)
    {
        LocalTensor<T> result = z_que.DeQue<T>();
#if __CCE_AICORE__ == 220
        DataCopyParams copyParams;
        copyParams.blockLen = length * sizeof(T);
        copyParams.blockCount = 1;
        DataCopyPad(z_gm[offset], result, copyParams);
#else
        int32_t blockNum = length / blockNumEl;
        int32_t tail = length % blockNumEl;
        if (blockNum != 0) {
            int32_t blkLength = blockNum * blockNumEl;
            DataCopy(z_gm[offset], result, blkLength);
            if (tail != 0) {
                SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
                for (uint32_t i = 0; i < blockNumEl; i++) {
                    T tensorValue = result.GetValue(length - blockNumEl + i);
                    result.SetValue(i, tensorValue);
                }
                SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
                DataCopy(z_gm[offset + length - blockNumEl], result, blockNumEl);
            }
        }
#endif
        z_que.FreeTensor(result);
    }

    __aicore__ inline void CommonCopyOutParam(int32_t offset)
    {
        LocalTensor<float> mean = mean_que_fp32.DeQue<float>();
        LocalTensor<float> rstd = rstd_que_fp32.DeQue<float>();

#if __CCE_AICORE__ == 220
        DataCopyParams copyParams;

        copyParams.blockLen = sizeof(float);
        copyParams.blockCount = 1;
        DataCopyPad(mean_gm[offset], mean, copyParams);

        copyParams.blockLen = sizeof(float);
        copyParams.blockCount = 1;
        DataCopyPad(rstd_gm[offset], rstd, copyParams);
#endif

        mean_que_fp32.FreeTensor(mean);
        rstd_que_fp32.FreeTensor(rstd);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> gx_que;
    TQue<QuePosition::VECIN, BUFFER_NUM> x_que;
    TQue<QuePosition::VECIN, BUFFER_NUM> beta_que;
    TQue<QuePosition::VECIN, BUFFER_NUM> gamma_que;

    TQue<QuePosition::VECOUT, BUFFER_NUM> z_que;
    TQue<QuePosition::VECOUT, BUFFER_NUM> mean_que_fp32;
    TQue<QuePosition::VECOUT, BUFFER_NUM> rstd_que_fp32;

    TBuf<TPosition::VECCALC> x_buf_fp32;
    TBuf<TPosition::VECCALC> y_buf_fp32;
    TBuf<TPosition::VECCALC> z_buf_fp32;
    TBuf<TPosition::VECCALC> calc_buf_fp32;
    TBuf<TPosition::VECCALC> calc_x_fp32;
    TBuf<TPosition::VECCALC> calc_y_fp32;

    GlobalTensor<T> x_gm;
    GlobalTensor<T> gx_gm;
    GlobalTensor<T> beta_gm;
    GlobalTensor<T> gamma_gm;
    GlobalTensor<T> z_gm;
    GlobalTensor<float> mean_gm;
    GlobalTensor<float> rstd_gm;
    int32_t databyte;
    uint32_t num_core;
    uint32_t num_first_dim;
    uint32_t num_last_dim;
    uint32_t row_work;
    uint32_t row_step;
    uint32_t row_tail_;
    uint32_t gm_offset_;
    uint32_t gm_offset2_;
    uint32_t nl_first_dim_per_core;
    uint32_t l_first_dim_per_core;
    uint32_t first_dim_per_times;
    uint32_t updated_last_dim;
    uint32_t updated_last_times;
    uint32_t blockNumEl = 0;
    float meanNum;
    float eps;
    float alphaVal;
    float meanVal;
    float varVal;
};

#if __CCE_AICORE__ != 220
#define bfloat16_t int16_t
#endif

extern "C" __global__ __aicore__ void deep_norm(GM_ADDR x, GM_ADDR gx, GM_ADDR beta, GM_ADDR gamma, GM_ADDR mean,
    GM_ADDR rstd, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    if (TILING_KEY_IS(0)) {  // bf16 && D <= 4096
        KernelDeepNorm<bfloat16_t> op;
        op.Init(x,
            gx,
            beta,
            gamma,
            mean,
            rstd,
            y,
            tiling_data.num_core,
            tiling_data.num_last_dim,
            tiling_data.num_first_dim,
            tiling_data.nl_firstdim_per_core,
            tiling_data.l_firstdim_per_core,
            tiling_data.first_dim_per_times,
            tiling_data.updated_last_dim,
            tiling_data.updated_last_times,
            tiling_data.eps_str,
            tiling_data.ave_str,
            tiling_data.alpha_str);
        op.ProcessBF16LELimit();
    } else if (TILING_KEY_IS(1)) {  // fp16 && D <= 4096
        KernelDeepNorm<half> op;
        op.Init(x,
            gx,
            beta,
            gamma,
            mean,
            rstd,
            y,
            tiling_data.num_core,
            tiling_data.num_last_dim,
            tiling_data.num_first_dim,
            tiling_data.nl_firstdim_per_core,
            tiling_data.l_firstdim_per_core,
            tiling_data.first_dim_per_times,
            tiling_data.updated_last_dim,
            tiling_data.updated_last_times,
            tiling_data.eps_str,
            tiling_data.ave_str,
            tiling_data.alpha_str);
        op.ProcessFp16LELimit();
    } else if (TILING_KEY_IS(2)) {  // fp32 && D <= 4096
        KernelDeepNorm<float> op;
        op.Init(x,
            gx,
            beta,
            gamma,
            mean,
            rstd,
            y,
            tiling_data.num_core,
            tiling_data.num_last_dim,
            tiling_data.num_first_dim,
            tiling_data.nl_firstdim_per_core,
            tiling_data.l_firstdim_per_core,
            tiling_data.first_dim_per_times,
            tiling_data.updated_last_dim,
            tiling_data.updated_last_times,
            tiling_data.eps_str,
            tiling_data.ave_str,
            tiling_data.alpha_str);
        op.ProcessFp32LELimit();
    } else if (TILING_KEY_IS(4)) {  // bf16 && D > 4096
        KernelDeepNorm<bfloat16_t> op;
        op.InitExtra(x,
            gx,
            beta,
            gamma,
            mean,
            rstd,
            y,
            tiling_data.num_core,
            tiling_data.num_last_dim,
            tiling_data.num_first_dim,
            tiling_data.nl_firstdim_per_core,
            tiling_data.l_firstdim_per_core,
            tiling_data.first_dim_per_times,
            tiling_data.updated_last_dim,
            tiling_data.updated_last_times,
            tiling_data.eps_str,
            tiling_data.ave_str,
            tiling_data.alpha_str);
        op.ProcessBf16GTLimit();
    } else if (TILING_KEY_IS(5)) {  // fp16 && D > 4096
        KernelDeepNorm<half> op;
        op.InitExtra(x,
            gx,
            beta,
            gamma,
            mean,
            rstd,
            y,
            tiling_data.num_core,
            tiling_data.num_last_dim,
            tiling_data.num_first_dim,
            tiling_data.nl_firstdim_per_core,
            tiling_data.l_firstdim_per_core,
            tiling_data.first_dim_per_times,
            tiling_data.updated_last_dim,
            tiling_data.updated_last_times,
            tiling_data.eps_str,
            tiling_data.ave_str,
            tiling_data.alpha_str);
        op.ProcessFp16GTLimit();
    } else if (TILING_KEY_IS(6)) {  // fp32 && D > 4096
        KernelDeepNorm<float> op;
        op.InitExtra(x,
            gx,
            beta,
            gamma,
            mean,
            rstd,
            y,
            tiling_data.num_core,
            tiling_data.num_last_dim,
            tiling_data.num_first_dim,
            tiling_data.nl_firstdim_per_core,
            tiling_data.l_firstdim_per_core,
            tiling_data.first_dim_per_times,
            tiling_data.updated_last_dim,
            tiling_data.updated_last_times,
            tiling_data.eps_str,
            tiling_data.ave_str,
            tiling_data.alpha_str);
        op.ProcessFp32GTLimit();
    } else if (TILING_KEY_IS(12)) {  // bf16 && D in common (>15360)
        KernelDeepNorm<bfloat16_t> op;
        op.InitCommon(x,
            gx,
            beta,
            gamma,
            mean,
            rstd,
            y,
            tiling_data.num_core,
            tiling_data.num_last_dim,
            tiling_data.num_first_dim,
            tiling_data.nl_firstdim_per_core,
            tiling_data.l_firstdim_per_core,
            tiling_data.first_dim_per_times,
            tiling_data.updated_last_dim,
            tiling_data.updated_last_times,
            tiling_data.eps_str,
            tiling_data.ave_str,
            tiling_data.alpha_str);
        op.ProcessBf16Common();
    } else if (TILING_KEY_IS(13)) {  // fp16 && D in common (>15360)
        KernelDeepNorm<half> op;
        op.InitCommon(x,
            gx,
            beta,
            gamma,
            mean,
            rstd,
            y,
            tiling_data.num_core,
            tiling_data.num_last_dim,
            tiling_data.num_first_dim,
            tiling_data.nl_firstdim_per_core,
            tiling_data.l_firstdim_per_core,
            tiling_data.first_dim_per_times,
            tiling_data.updated_last_dim,
            tiling_data.updated_last_times,
            tiling_data.eps_str,
            tiling_data.ave_str,
            tiling_data.alpha_str);
        op.ProcessFp16Common();
    } else if (TILING_KEY_IS(14)) {  // fp32 && D in common (>8192)
        KernelDeepNorm<float> op;
        op.InitCommon(x,
            gx,
            beta,
            gamma,
            mean,
            rstd,
            y,
            tiling_data.num_core,
            tiling_data.num_last_dim,
            tiling_data.num_first_dim,
            tiling_data.nl_firstdim_per_core,
            tiling_data.l_firstdim_per_core,
            tiling_data.first_dim_per_times,
            tiling_data.updated_last_dim,
            tiling_data.updated_last_times,
            tiling_data.eps_str,
            tiling_data.ave_str,
            tiling_data.alpha_str);
        op.ProcessFp32Common();
    } else if (TILING_KEY_IS(16)) {  // bf16 && D <= 500
        KernelDeepNorm<bfloat16_t> op;
        op.InitShort(x,
            gx,
            beta,
            gamma,
            mean,
            rstd,
            y,
            tiling_data.num_core,
            tiling_data.num_last_dim,
            tiling_data.num_first_dim,
            tiling_data.nl_firstdim_per_core,
            tiling_data.l_firstdim_per_core,
            tiling_data.first_dim_per_times,
            tiling_data.updated_last_dim,
            tiling_data.updated_last_times,
            tiling_data.eps_str,
            tiling_data.ave_str,
            tiling_data.alpha_str);
        op.ProcessBf16Short();
    } else if (TILING_KEY_IS(17)) {  // fp16 && D <= 500
        KernelDeepNorm<half> op;
        op.InitShort(x,
            gx,
            beta,
            gamma,
            mean,
            rstd,
            y,
            tiling_data.num_core,
            tiling_data.num_last_dim,
            tiling_data.num_first_dim,
            tiling_data.nl_firstdim_per_core,
            tiling_data.l_firstdim_per_core,
            tiling_data.first_dim_per_times,
            tiling_data.updated_last_dim,
            tiling_data.updated_last_times,
            tiling_data.eps_str,
            tiling_data.ave_str,
            tiling_data.alpha_str);
        op.ProcessFp16Short();
    } else if (TILING_KEY_IS(18)) {  // fp32 && D <= 500
        KernelDeepNorm<float> op;
        op.InitShort(x,
            gx,
            beta,
            gamma,
            mean,
            rstd,
            y,
            tiling_data.num_core,
            tiling_data.num_last_dim,
            tiling_data.num_first_dim,
            tiling_data.nl_firstdim_per_core,
            tiling_data.l_firstdim_per_core,
            tiling_data.first_dim_per_times,
            tiling_data.updated_last_dim,
            tiling_data.updated_last_times,
            tiling_data.eps_str,
            tiling_data.ave_str,
            tiling_data.alpha_str);
        op.ProcessFp32Short();
    }
}