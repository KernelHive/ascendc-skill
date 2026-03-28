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
 * \file deep_norm_grad_merge_n.h
 * \brief
 */
#ifndef OPS_BUILT_IN_TBE_IMPL_ASCENDC_DEEP_NORM_GRAD_DEEP_NORM_GRAD_MERGE_N_H_
#define OPS_BUILT_IN_TBE_IMPL_ASCENDC_DEEP_NORM_GRAD_DEEP_NORM_GRAD_MERGE_N_H_
#include "deep_norm_grad_common.h"

template <typename T>
class KernelDeepNormGradMergeN : public KernelDeepNormGradBase<T> {
public:
    __aicore__ inline KernelDeepNormGradMergeN()
    {}
    __aicore__ inline void InitGM(GM_ADDR data_dy, GM_ADDR data_x, GM_ADDR data_gx, GM_ADDR data_rstd,
        GM_ADDR data_mean, GM_ADDR data_gamma, GM_ADDR output_pd_x, GM_ADDR output_pd_gx, GM_ADDR output_pd_gamma,
        GM_ADDR output_pd_beta)
    {
        dy_gm.SetGlobalBuffer((__gm__ T *)data_dy + GetBlockIdx() * N_deal_per_core * D_dim_num, N_deal * D_dim_num);
        x_gm.SetGlobalBuffer((__gm__ T *)data_x + GetBlockIdx() * N_deal_per_core * D_dim_num, N_deal * D_dim_num);
        gx_gm.SetGlobalBuffer((__gm__ T *)data_gx + GetBlockIdx() * N_deal_per_core * D_dim_num, N_deal * D_dim_num);
        mean_gm.SetGlobalBuffer((__gm__ float *)data_mean + GetBlockIdx() * N_deal_per_core, N_deal);
        rstd_gm.SetGlobalBuffer((__gm__ float *)data_rstd + GetBlockIdx() * N_deal_per_core, N_deal);
        gamma_gm.SetGlobalBuffer((__gm__ T *)data_gamma, D_dim_num);

        output_pd_x_gm.SetGlobalBuffer(
            (__gm__ T *)output_pd_x + GetBlockIdx() * N_deal_per_core * D_dim_num, N_deal * D_dim_num);
        output_pd_gx_gm.SetGlobalBuffer(
            (__gm__ T *)output_pd_gx + GetBlockIdx() * N_deal_per_core * D_dim_num, N_deal * D_dim_num);
        output_pd_beta_gm.SetGlobalBuffer((__gm__ float *)output_pd_beta, D_dim_num);
        output_pd_gamma_gm.SetGlobalBuffer((__gm__ float *)output_pd_gamma, D_dim_num);
    }

    __aicore__ inline void InitQueue()
    {
        pipe.InitBuffer(dy_que, BUFFER_NUM, merge_N_count_update_per * elem_with_D_in_ub * sizeof(T));
        pipe.InitBuffer(x_que, BUFFER_NUM, merge_N_count_update_per * elem_with_D_in_ub * sizeof(T));
        pipe.InitBuffer(gx_que, BUFFER_NUM, merge_N_count_update_per * elem_with_D_in_ub * sizeof(T));
        pipe.InitBuffer(mean_que, BUFFER_NUM, merge_N_count_update_per * elem_without_D_in_ub_fp32 * sizeof(float));
        pipe.InitBuffer(rstd_que, BUFFER_NUM, merge_N_count_update_per * elem_without_D_in_ub_fp32 * sizeof(float));
        pipe.InitBuffer(gamma_que, BUFFER_NUM, elem_with_D_in_ub * sizeof(T));
        pipe.InitBuffer(output_pd_x_que, BUFFER_NUM, merge_N_count_update_per * elem_with_D_in_ub * sizeof(T));
        pipe.InitBuffer(output_pd_gx_que, BUFFER_NUM, merge_N_count_update_per * elem_with_D_in_ub * sizeof(T));
        pipe.InitBuffer(output_pd_beta_que, BUFFER_NUM, elem_with_D_in_ub_fp32 * sizeof(float));
        pipe.InitBuffer(output_pd_gamma_que, BUFFER_NUM, elem_with_D_in_ub_fp32 * sizeof(float));
#if __CCE_AICORE__ == 220
        if constexpr (IsSame<T, half>::value || IsSame<T, bfloat16_t>::value) {
#else
        if constexpr (IsSame<T, half>::value) {
#endif
            pipe.InitBuffer(dy_fp32_buf, elem_with_D_in_ub_fp32 * sizeof(float));
            pipe.InitBuffer(x_fp32_buf, elem_with_D_in_ub_fp32 * sizeof(float));
            pipe.InitBuffer(gx_fp32_buf, elem_with_D_in_ub_fp32 * sizeof(float));
            pipe.InitBuffer(gamma_fp32_buf, elem_with_D_in_ub_fp32 * sizeof(float));
            pipe.InitBuffer(output_pd_x_fp32_buf, elem_with_D_in_ub_fp32 * sizeof(float));
            pipe.InitBuffer(output_pd_gx_fp32_buf, elem_with_D_in_ub_fp32 * sizeof(float));
        }
    }

    __aicore__ inline void Init(GM_ADDR data_dy, GM_ADDR data_x, GM_ADDR data_gx, GM_ADDR data_rstd, GM_ADDR data_mean,
        GM_ADDR data_gamma, GM_ADDR output_pd_x, GM_ADDR output_pd_gx, GM_ADDR output_pd_gamma, GM_ADDR output_pd_beta,
        DeepNormGradTilingData tiling, GM_ADDR usrWorkspace)
    {
        use_core_num = tiling.useCoreNum;
        N_dim_num = tiling.nDimNum;
        D_dim_num = tiling.dDimNum;
        N_deal_per_core = tiling.nDealPerCore;
        N_deal_last_core = tiling.nDealLastCore;
        alpha_val = *reinterpret_cast<float *>(&tiling.alpha);
        fixedOutputFlag = tiling.fixedOutputFlag;

        merge_N_count = tiling.mergeNCount;  // >1: no cut;

        // init GM
        N_deal = (GetBlockIdx() != use_core_num - 1) ? N_deal_per_core : N_deal_last_core;
        InitGM(data_dy,
            data_x,
            data_gx,
            data_rstd,
            data_mean,
            data_gamma,
            output_pd_x,
            output_pd_gx,
            output_pd_gamma,
            output_pd_beta);

        // merge N
        merge_N_count_update_per = (merge_N_count > N_deal) ? N_deal : merge_N_count;
        merge_N_time = this->CeilDiv(N_deal, merge_N_count_update_per);
        merge_N_count_update_tail = N_deal - (merge_N_count_update_per * (merge_N_time - 1));

        // cut D
        uint32_t D_dim_num_alloc = D_dim_num;

        // init queue
        block_elem = BLOCK_ALIGN_SIZE / sizeof(T);
        block_elem_fp32 = BLOCK_ALIGN_SIZE / sizeof(float);

        elem_with_D_in_ub = this->BlockAlign(D_dim_num_alloc, block_elem);
        elem_without_D_in_ub = this->BlockAlign(1, block_elem);
        elem_with_D_in_ub_fp32 = this->BlockAlign(D_dim_num_alloc, block_elem_fp32);
        elem_without_D_in_ub_fp32 = this->BlockAlign(1, block_elem_fp32);

        InitQueue();

        // use atomicadd, need init beta&gamma
        LocalTensor<float> temp_local_tensor = output_pd_gamma_que.AllocTensor<float>();
        this->InitGmData(
            output_pd_gamma_gm, output_pd_beta_gm, D_dim_num, temp_local_tensor, elem_without_D_in_ub_fp32);
        output_pd_gamma_que.FreeTensor(temp_local_tensor);

        // avoid muti cal in UB
        one_div_D = (float)-1.0 / D_dim_num;

#if __CCE_AICORE__ == 220
        SyncAll();
#else
        uint32_t each_core_handle_num = BLOCK_ALIGN_SIZE / sizeof(int32_t);
        GlobalTensor<int32_t> syncGlobal_;
        syncGlobal_.SetGlobalBuffer((__gm__ int32_t *)usrWorkspace, use_core_num * block_elem_fp32);

        LocalTensor<int32_t> tmp_init_buf = output_pd_beta_que.AllocTensor<int32_t>();
        Duplicate(tmp_init_buf, 0, each_core_handle_num);
        DataCopy(syncGlobal_[each_core_handle_num * GetBlockIdx()], tmp_init_buf, each_core_handle_num);

        LocalTensor<int32_t> workLocal = output_pd_gamma_que.AllocTensor<int32_t>();
        SyncAll(syncGlobal_, workLocal);
        output_pd_gamma_que.FreeTensor(workLocal);
        output_pd_beta_que.FreeTensor(tmp_init_buf);

#endif
    }

    __aicore__ inline void Process()
    {
        CopyInPre(D_dim_num);
        LocalTensor<T> gamma = gamma_que.DeQue<T>();
        LocalTensor<float> dbeta = output_pd_beta_que.AllocTensor<float>();
        LocalTensor<float> dgamma = output_pd_gamma_que.AllocTensor<float>();

        // init atomic Tensor
        Duplicate(dbeta, 0.0f, elem_with_D_in_ub_fp32);
        Duplicate(dgamma, 0.0f, elem_with_D_in_ub_fp32);
#if __CCE_AICORE__ == 220
        if constexpr (IsSame<T, half>::value || IsSame<T, bfloat16_t>::value) {
#else
        if constexpr (IsSame<T, half>::value) {
#endif
            LocalTensor<float> gamma_fp32 = gamma_fp32_buf.Get<float>();
            Cast(gamma_fp32, gamma, RoundMode::CAST_NONE, D_dim_num);
            ProcessMergeN(gamma_fp32, dbeta, dgamma);
        } else {
            ProcessMergeN(gamma, dbeta, dgamma);
        }

        gamma_que.FreeTensor(gamma);
        output_pd_beta_que.EnQue(dbeta);
        output_pd_gamma_que.EnQue(dgamma);

        if (fixedOutputFlag == 0) {
            CopyOutAfter(D_dim_num);
        } else {
            CopyOutAfterInOrder(D_dim_num);
        }
    }

    __aicore__ inline void ProcessMergeN(
        const LocalTensor<float> &gamma, const LocalTensor<float> &dbeta, const LocalTensor<float> &dgamma)
    {
        for (uint32_t i_merge = 0; i_merge < merge_N_time; ++i_merge) {
            uint32_t merge_N_count_update =
                (i_merge != merge_N_time - 1) ? merge_N_count_update_per : merge_N_count_update_tail;

            CopyInMergeN(i_merge, merge_N_count_update, D_dim_num);
            ComputeMergeN(i_merge, merge_N_count_update, D_dim_num, gamma, dbeta, dgamma);
            CopyOutMergeN(i_merge, merge_N_count_update, D_dim_num);
        }
    }

private:
    __aicore__ inline void CopyInPre(uint32_t process_elem)
    {
        LocalTensor<T> gamma_local = gamma_que.AllocTensor<T>();
        uint32_t offset_D = 0;
#if __CCE_AICORE__ == 220
        // gamma
        DataCopyParams data_copy_params_D{(uint16_t)1, (uint16_t)(process_elem * sizeof(T)), 0, 0};
        uint8_t right_pad_elem_D = this->BlockAlign(process_elem, block_elem) - process_elem;
        DataCopyPadParams right_pad_params_D{true, 0, right_pad_elem_D, 0};

        DataCopyPad(gamma_local, gamma_gm[offset_D], data_copy_params_D, right_pad_params_D);
#else
        DataCopy(gamma_local, gamma_gm[offset_D], this->BlockAlign(process_elem, block_elem));
#endif
        gamma_que.EnQue(gamma_local);
    }

    __aicore__ inline void CopyOutAfter(uint32_t process_elem)
    {
        LocalTensor<float> dbeta = output_pd_beta_que.DeQue<float>();
        LocalTensor<float> dgamma = output_pd_gamma_que.DeQue<float>();

        SetAtomicAdd<float>();
        DataCopyAutomicAdd(output_pd_gamma_gm, dgamma, process_elem, 0, (uint16_t)1);
        DataCopyAutomicAdd(output_pd_beta_gm, dbeta, process_elem, 0, (uint16_t)1);
        SetAtomicNone();

        output_pd_beta_que.FreeTensor(dbeta);
        output_pd_gamma_que.FreeTensor(dgamma);
    }

    __aicore__ inline void CopyOutAfterInOrder(uint32_t process_elem)
    {
        uint32_t alreadyFixOutputSyncVal = GetBlockIdx();
        for (int32_t count = 0; count < INT_MAX; count++) {
            if (g_FixedOutputSync[0] == alreadyFixOutputSyncVal) {
                break;
            }
        }

        LocalTensor<float> dbeta = output_pd_beta_que.DeQue<float>();
        LocalTensor<float> dgamma = output_pd_gamma_que.DeQue<float>();

        SetAtomicAdd<float>();
        DataCopyAutomicAdd(output_pd_gamma_gm, dgamma, process_elem, 0, (uint16_t)1);
        DataCopyAutomicAdd(output_pd_beta_gm, dbeta, process_elem, 0, (uint16_t)1);
        SetAtomicNone();

        event_t eventMTE3S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(eventMTE3S);
        WaitFlag<HardEvent::MTE3_S>(eventMTE3S);

        output_pd_beta_que.FreeTensor(dbeta);
        output_pd_gamma_que.FreeTensor(dgamma);
        g_FixedOutputSync[0]++;
    }

    __aicore__ inline void CopyInMergeN(uint32_t process_id, uint32_t process_N_count, uint32_t process_elem)
    {
        LocalTensor<T> dy_local = dy_que.AllocTensor<T>();
        LocalTensor<T> x_local = x_que.AllocTensor<T>();
        LocalTensor<T> gx_local = gx_que.AllocTensor<T>();
        LocalTensor<float> mean_local = mean_que.AllocTensor<float>();
        LocalTensor<float> rstd_local = rstd_que.AllocTensor<float>();

        uint32_t offset_ND = process_id * merge_N_count_update_per * D_dim_num;
        uint32_t offset_N = process_id * merge_N_count_update_per;

#if __CCE_AICORE__ == 220
        // dy&x&gx
        DataCopyParams data_copy_params_ND{(uint16_t)process_N_count, (uint16_t)(process_elem * sizeof(T)), 0, 0};
        uint8_t right_pad_elem_ND = this->BlockAlign(process_elem, block_elem) - process_elem;
        DataCopyPadParams right_pad_params_ND{true, 0, right_pad_elem_ND, 0};

        DataCopyPad(dy_local, dy_gm[offset_ND], data_copy_params_ND, right_pad_params_ND);
        DataCopyPad(x_local, x_gm[offset_ND], data_copy_params_ND, right_pad_params_ND);
        DataCopyPad(gx_local, gx_gm[offset_ND], data_copy_params_ND, right_pad_params_ND);

        // mean&rstd
        DataCopyParams data_copy_params_N{(uint16_t)process_N_count, (uint16_t)(1 * sizeof(float)), 0, 0};
        uint8_t right_pad_elem_N = this->BlockAlign(1, block_elem_fp32) - 1;
        DataCopyPadParams right_pad_params_N{true, 0, right_pad_elem_N, 0};

        DataCopyPad(mean_local, mean_gm[offset_N], data_copy_params_N, right_pad_params_N);
        DataCopyPad(rstd_local, rstd_gm[offset_N], data_copy_params_N, right_pad_params_N);
#else
        for (uint32_t idx = 0; idx < process_N_count; idx++) {
            DataCopy(dy_local[idx * this->BlockAlign(process_elem, block_elem)],
                dy_gm[offset_ND + idx * process_elem],
                this->BlockAlign(process_elem, block_elem));
            DataCopy(x_local[idx * this->BlockAlign(process_elem, block_elem)],
                x_gm[offset_ND + idx * process_elem],
                this->BlockAlign(process_elem, block_elem));
            DataCopy(gx_local[idx * this->BlockAlign(process_elem, block_elem)],
                gx_gm[offset_ND + idx * process_elem],
                this->BlockAlign(process_elem, block_elem));

            DataCopy(mean_local[idx * this->BlockAlign(1, block_elem_fp32)],
                mean_gm[offset_N + idx],
                this->BlockAlign(1, block_elem_fp32));
            DataCopy(rstd_local[idx * this->BlockAlign(1, block_elem_fp32)],
                rstd_gm[offset_N + idx],
                this->BlockAlign(1, block_elem_fp32));
        }
#endif
        dy_que.EnQue(dy_local);
        x_que.EnQue(x_local);
        gx_que.EnQue(gx_local);
        mean_que.EnQue(mean_local);
        rstd_que.EnQue(rstd_local);
    }

    __aicore__ inline void CopyOutMergeN(uint32_t process_id, uint32_t process_N_count, uint32_t process_elem)
    {
        LocalTensor<T> output_pd_x_local = output_pd_x_que.DeQue<T>();
        LocalTensor<T> output_pd_gx_local = output_pd_gx_que.DeQue<T>();

        uint32_t offset_ND = process_id * merge_N_count_update_per * D_dim_num;

        DataCopyCustom<T>(output_pd_x_gm, output_pd_x_local, process_elem, offset_ND, false, (uint16_t)process_N_count);
        DataCopyCustom<T>(
            output_pd_gx_gm, output_pd_gx_local, process_elem, offset_ND, false, (uint16_t)process_N_count);

        output_pd_x_que.FreeTensor(output_pd_x_local);
        output_pd_gx_que.FreeTensor(output_pd_gx_local);
    }

    __aicore__ inline void ComputeMergeN(uint32_t process_id, uint32_t process_N_count, uint32_t process_elem,
        const LocalTensor<float> &gamma, const LocalTensor<float> &dbeta, const LocalTensor<float> &dgamma)
    {
        LocalTensor<T> inputDy = dy_que.DeQue<T>();
        LocalTensor<T> inputX = x_que.DeQue<T>();
        LocalTensor<T> inputGx = gx_que.DeQue<T>();
        LocalTensor<float> inputRstd = rstd_que.DeQue<float>();
        LocalTensor<float> inputMean = mean_que.DeQue<float>();

        LocalTensor<T> outputPdX = output_pd_x_que.AllocTensor<T>();
        LocalTensor<T> outputPdGx = output_pd_gx_que.AllocTensor<T>();

        for (uint32_t N_index = 0; N_index < process_N_count; ++N_index) {
            MainComputeWithCast(inputDy,
                inputX,
                inputGx,
                inputRstd,
                inputMean,
                gamma,
                outputPdX,
                outputPdGx,
                dbeta,
                dgamma,
                process_elem,
                N_index);
        }

        dy_que.FreeTensor(inputDy);
        x_que.FreeTensor(inputX);
        gx_que.FreeTensor(inputGx);
        rstd_que.FreeTensor(inputRstd);
        mean_que.FreeTensor(inputMean);
        output_pd_x_que.EnQue(outputPdX);
        output_pd_gx_que.EnQue(outputPdGx);
    }

    __aicore__ inline void MainComputeWithCast(const LocalTensor<T> &inputDy, const LocalTensor<T> &inputX,
        const LocalTensor<T> &inputGx, const LocalTensor<float> &inputRstd, const LocalTensor<float> &inputMean,
        const LocalTensor<float> &inputGamma, const LocalTensor<T> &outputPdX, const LocalTensor<T> &outputPdGx,
        const LocalTensor<float> &outputPdBeta, const LocalTensor<float> &outputPdGamma, uint32_t process_elem,
        uint32_t N_index)
    {
        uint32_t offset_ND_in_ub = N_index * elem_with_D_in_ub;
        uint32_t offset_N_in_ub_fp32 = N_index * elem_without_D_in_ub_fp32;

#if __CCE_AICORE__ == 220
        if constexpr (IsSame<T, half>::value || IsSame<T, bfloat16_t>::value) {
#else
        if constexpr (IsSame<T, half>::value) {
#endif
            LocalTensor<float> dy_fp32_local = dy_fp32_buf.Get<float>();
            LocalTensor<float> x_fp32_local = x_fp32_buf.Get<float>();
            LocalTensor<float> gx_fp32_local = gx_fp32_buf.Get<float>();
            LocalTensor<float> output_pd_x_fp32_local = output_pd_x_fp32_buf.Get<float>();
            LocalTensor<float> output_pd_gx_fp32_local = output_pd_gx_fp32_buf.Get<float>();
            Cast(dy_fp32_local, inputDy[offset_ND_in_ub], RoundMode::CAST_NONE, process_elem);
            Cast(x_fp32_local, inputX[offset_ND_in_ub], RoundMode::CAST_NONE, process_elem);
            Cast(gx_fp32_local, inputGx[offset_ND_in_ub], RoundMode::CAST_NONE, process_elem);
            PipeBarrier<PIPE_V>();

            MainCompute(dy_fp32_local,
                x_fp32_local,
                gx_fp32_local,
                inputRstd[offset_N_in_ub_fp32],
                inputMean[offset_N_in_ub_fp32],
                inputGamma,
                output_pd_x_fp32_local,
                output_pd_gx_fp32_local,
                outputPdBeta,
                outputPdGamma,
                process_elem);

            if constexpr (IsSame<T, half>::value) {
                Cast(outputPdX[offset_ND_in_ub], output_pd_x_fp32_local, RoundMode::CAST_NONE, process_elem);
                Cast(outputPdGx[offset_ND_in_ub], output_pd_gx_fp32_local, RoundMode::CAST_NONE, process_elem);
            } else {
                Cast(outputPdX[offset_ND_in_ub], output_pd_x_fp32_local, RoundMode::CAST_RINT, process_elem);
                Cast(outputPdGx[offset_ND_in_ub], output_pd_gx_fp32_local, RoundMode::CAST_RINT, process_elem);
            }
        } else {
            MainCompute(inputDy[offset_ND_in_ub],
                inputX[offset_ND_in_ub],
                inputGx[offset_ND_in_ub],
                inputRstd[offset_N_in_ub_fp32],
                inputMean[offset_N_in_ub_fp32],
                inputGamma,
                outputPdX[offset_ND_in_ub],
                outputPdGx[offset_ND_in_ub],
                outputPdBeta,
                outputPdGamma,
                process_elem);
        }
    }

    __aicore__ inline void MainCompute(const LocalTensor<float> &inputDy, const LocalTensor<float> &inputX1,
        const LocalTensor<float> &inputX2, const LocalTensor<float> &inputRstd, const LocalTensor<float> &inputMean,
        const LocalTensor<float> &inputGamma, const LocalTensor<float> &outputPdX, const LocalTensor<float> &outputPdGx,
        const LocalTensor<float> &outputPdBeta, const LocalTensor<float> &outputPdGamma, uint32_t process_elem)
    {
        // 0. x_sum = alpha * x1 + x2
        Axpy(inputX2, inputX1, alpha_val, process_elem);
        PipeBarrier<PIPE_V>();

        // 1. x1Tensor = dy * gamma
        Mul(inputX1, inputDy, inputGamma, process_elem);

        event_t event_mte2_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));

        // 2. x2Tensor = x_sum - mean
        SetFlag<HardEvent::MTE2_S>(event_mte2_s);
        WaitFlag<HardEvent::MTE2_S>(event_mte2_s);
        float input_mean_num = inputMean.GetValue(0);
        float input_rstd_num = inputRstd.GetValue(0);
        float rstd_sqrt_tmp_num = input_rstd_num * input_rstd_num * input_rstd_num;
        SetFlag<HardEvent::S_V>(event_s_v);
        WaitFlag<HardEvent::S_V>(event_s_v);

        Adds(inputX2, inputX2, input_mean_num * (-1.0f), process_elem);
        PipeBarrier<PIPE_V>();

        // 3. d_var = sum((-0.5) * x1Tensor * x2Tensor * np.power(inputRstd, 3))
        // 3.1. tmp = (-0.5) * x1Tensor * x2Tensor * rstd^3
        Muls(outputPdGx, inputX2, rstd_sqrt_tmp_num, process_elem);
        PipeBarrier<PIPE_V>();
        Mul(outputPdGx, outputPdGx, inputX1, process_elem);
        PipeBarrier<PIPE_V>();
        // 3.2. d_var = sum(tmp)
        auto reduce_tmp_num = this->ReduceSumCustom(outputPdGx, process_elem);
        input_mean_num = reduce_tmp_num * one_div_D;
        SetFlag<HardEvent::S_V>(event_s_v);
        WaitFlag<HardEvent::S_V>(event_s_v);

        // other (2 / D * d_var * x2Tensor)
        Muls(outputPdGx, inputX2, input_mean_num, process_elem);

        // 4. d_mean = np.sum( (-1.0) * x1Tensor * rstd) )
        // 4.1. tmp1 = (-1.0) * x1Tensor * rstd
        Muls(outputPdX, inputX1, input_rstd_num, process_elem);  // use in d_gx cal
        PipeBarrier<PIPE_V>();

        // other: (x2Tensor * rstd) + (2 / D * d_var * x1Tensor)
        Add(outputPdGx, outputPdGx, outputPdX, process_elem);

        // 4.2. d_mean = np.sum(tmp1)
        reduce_tmp_num = this->ReduceSumCustom(outputPdX, process_elem);
        input_mean_num = reduce_tmp_num * one_div_D;
        PipeBarrier<PIPE_V>();

        // 5. d_gx = x2Tensor * rstd + d_var * (2.0 / D) * x1Tensor (already)
        //           + d_mean * (1.0 / D)
        Adds(outputPdGx, outputPdGx, input_mean_num, process_elem);
        PipeBarrier<PIPE_V>();

        Muls(outputPdX, outputPdGx, alpha_val, process_elem);
        PipeBarrier<PIPE_V>();

        // 6. d_gamma_part = add ( x2Tensor * rstd * dy )
        Muls(inputX2, inputX2, input_rstd_num, process_elem);
        PipeBarrier<PIPE_V>();
        Mul(inputX2, inputX2, inputDy, process_elem);
        PipeBarrier<PIPE_V>();
        Add(outputPdGamma, outputPdGamma, inputX2, process_elem);

        // 7. d_beta_part = add ( dy )
        Add(outputPdBeta, outputPdBeta, inputDy, process_elem);
    }

public:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> dy_que;
    TQue<QuePosition::VECIN, BUFFER_NUM> x_que;
    TQue<QuePosition::VECIN, BUFFER_NUM> gx_que;
    TQue<QuePosition::VECIN, BUFFER_NUM> rstd_que;
    TQue<QuePosition::VECIN, BUFFER_NUM> mean_que;
    TQue<QuePosition::VECIN, BUFFER_NUM> gamma_que;
    TQue<QuePosition::VECOUT, BUFFER_NUM> output_pd_x_que;
    TQue<QuePosition::VECOUT, BUFFER_NUM> output_pd_gx_que;
    TQue<QuePosition::VECOUT, BUFFER_NUM> output_pd_beta_que;
    TQue<QuePosition::VECOUT, BUFFER_NUM> output_pd_gamma_que;

    // cast buf for fp16&bf16
    TBuf<TPosition::VECCALC> dy_fp32_buf;
    TBuf<TPosition::VECCALC> x_fp32_buf;
    TBuf<TPosition::VECCALC> gx_fp32_buf;
    TBuf<TPosition::VECCALC> gamma_fp32_buf;
    TBuf<TPosition::VECCALC> output_pd_x_fp32_buf;
    TBuf<TPosition::VECCALC> output_pd_gx_fp32_buf;
    TBuf<TPosition::VECCALC> x_buf_fp32;

    // input
    GlobalTensor<T> dy_gm, x_gm, gx_gm, gamma_gm;
    GlobalTensor<float> rstd_gm, mean_gm;

    // output
    GlobalTensor<T> output_pd_x_gm, output_pd_gx_gm;
    GlobalTensor<float> output_pd_gamma_gm, output_pd_beta_gm;

    uint32_t use_core_num;
    uint32_t N_dim_num;
    uint32_t D_dim_num;
    uint32_t N_deal_per_core;
    uint32_t N_deal_last_core;

    uint32_t N_deal;
    uint32_t block_elem;
    uint32_t block_elem_fp32;

    // merge N params
    uint32_t merge_N_count;
    uint32_t merge_N_count_update_per;
    uint32_t merge_N_count_update_tail;
    uint32_t merge_N_time;

    uint32_t elem_with_D_in_ub;
    uint32_t elem_without_D_in_ub;
    uint32_t elem_with_D_in_ub_fp32;
    uint32_t elem_without_D_in_ub_fp32;

    float one_div_D;
    float two_div_D;
    float alpha_val;
    uint32_t fixedOutputFlag;
};

#endif
