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
 * \file group_norm_grad.h
 * \brief
 */
#ifndef GROUP_NORM_GRAD_H
#define GROUP_NORM_GRAD_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "kernel_type.h"
using namespace AscendC;
template <typename T, bool isDeterministic>
class GroupNormGrad {
public:
    __aicore__ inline GroupNormGrad(GM_ADDR dy, GM_ADDR mean, GM_ADDR rstd, GM_ADDR x, GM_ADDR gamma, GM_ADDR dx,
        GM_ADDR dgamma, GM_ADDR dbeta, GM_ADDR workspace, GroupNormGradTilingData *tiling_data)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->curBlockIdx = GetBlockIdx();
        this->Tiling_key = tiling_data->Tiling_key;
        this->N = tiling_data->N;
        this->C = tiling_data->C;
        this->G = tiling_data->G;
        this->HXW = tiling_data->HXW;
        this->NXG = tiling_data->NXG;
        this->C_G = tiling_data->C_G;
        this->task_num_per_core = tiling_data->task_num_per_core;
        this->task_num_per_tail_core = tiling_data->task_num_per_tail_core;
        this->tail_core = tiling_data->tail_core;
        this->workSpaceSize = tiling_data->workSpaceSize;
        this->all_ele_num = N * C * HXW;
        this->ele_num_per_group = C_G * HXW;
        this->ele_num_per_channel = HXW;
        this->dxIsRequire = tiling_data->dx_is_require;
        this->dgammaIsRequire = tiling_data->dgamma_is_require;
        this->dbetaIsRequire = tiling_data->dbeta_is_require;
        this->stage2CoreUsed = tiling_data->stage2CoreUsed;
        this->castEleNum = tiling_data->castEleNum;
        this->tailCastNum = tiling_data->tailCastNum;
        this->coreBatchParts = tiling_data->coreBatchParts;
        this->cur_core_task_num = this->task_num_per_core;
        this->start_task_id = this->task_num_per_core * this->curBlockIdx;

        if (this->curBlockIdx >= this->tail_core) {
            this->cur_core_task_num = this->task_num_per_tail_core;
            this->start_task_id = this->task_num_per_tail_core * this->curBlockIdx + this->tail_core;
        }
        this->T_per_block = BLOCK_BYTES / sizeof(T);
        dy_gm.SetGlobalBuffer((__gm__ T *)dy, this->all_ele_num);
        mean_gm.SetGlobalBuffer((__gm__ T *)mean, this->NXG);
        rstd_gm.SetGlobalBuffer((__gm__ T *)rstd, this->NXG);
        x_gm.SetGlobalBuffer((__gm__ T *)x, this->all_ele_num);
        gamma_gm.SetGlobalBuffer((__gm__ T *)gamma, this->C);
        dx_gm.SetGlobalBuffer((__gm__ T *)dx, this->all_ele_num);
        dgamma_gm.SetGlobalBuffer((__gm__ T *)dgamma, this->C);
        dbeta_gm.SetGlobalBuffer((__gm__ T *)dbeta, this->C);
        dgamma_workspace.SetGlobalBuffer((__gm__ float *)workspace, this->workSpaceSize);
        dbeta_workspace.SetGlobalBuffer((__gm__ float *)workspace + this->workSpaceSize, this->workSpaceSize);
        // init output or workspace
        Init_Output_GM();
        // init unified buffer
        Init_Unified_Buffer(tiling_data);
#ifndef __CCE_KT_TEST__
        // wait core
        SyncAll();
#endif
    }

    __aicore__ inline void Process()
    {
        // tiling strategy, pipeline parallel
        if (Tiling_key == MODE_0) {
            for (int32_t task_idx = 0; task_idx < this->cur_core_task_num; task_idx++) {
                CopyIn_Mode_0(task_idx + this->start_task_id);
                Compute_Mode_0(task_idx + this->start_task_id);
            }
        } else if (Tiling_key == MODE_1) {
            for (int32_t task_idx = 0; task_idx < this->cur_core_task_num; task_idx++) {
                Get_Mean_Rstd(task_idx + this->start_task_id);
                Compute_Mode_1(task_idx + this->start_task_id);
            }
        } else if (Tiling_key == MODE_2) {
            for (int32_t task_idx = 0; task_idx < this->cur_core_task_num; task_idx++) {
                Get_Mean_Rstd(task_idx + this->start_task_id);
                Compute_Mode_2(task_idx + this->start_task_id);
            }
        } else if (Tiling_key == MODE_3) {
            for (int32_t task_idx = 0; task_idx < this->cur_core_task_num; task_idx++) {
                Get_Mean_Rstd(task_idx + this->start_task_id);
                Compute_Mode_3(task_idx + this->start_task_id);
            }
        } else if (Tiling_key == MODE_5) {
            for (int32_t task_idx = 0; task_idx < this->cur_core_task_num; task_idx++) {
                CopyIn_Mode_0(task_idx + this->start_task_id);
                Compute_Mode_5(task_idx + this->start_task_id);
            }
        }
        pipe.Reset();
        constexpr uint32_t SPLIT_COUNT = 2;
        if constexpr (!isDeterministic && !IsSameType<T, float>::value) {
#ifndef __CCE_KT_TEST__
            // wait core
            SyncAll();
#endif
            if (N != 1 && curBlockIdx < stage2CoreUsed) {
                Initbuffer4stage2_mode0();
                // cast TO GM
                if (curBlockIdx < stage2CoreUsed - 1) {
                    cast_dgamma_dbeta_WSP2GM(curBlockIdx * castEleNum, castEleNum);
                } else if (curBlockIdx == stage2CoreUsed - 1) {
                    cast_dgamma_dbeta_WSP2GM(curBlockIdx * castEleNum, tailCastNum);
                }
            }
        } else if constexpr (isDeterministic && IsSameType<T, float>::value) {
#ifndef __CCE_KT_TEST__
            // wait core
            SyncAll();
#endif
            Initbuffer4stage2_mode1();
            // reduceSUM TO GM
            //  0 ~ stage2CoreUsed - 2
            if (curBlockIdx < stage2CoreUsed - 1) {
                reduce_dgamma_dbeta_WSP2GM(
                    0, curBlockIdx * castEleNum, castEleNum, divceil(divceil(N, SPLIT_COUNT), SPLIT_COUNT));
            } else if (stage2CoreUsed <= curBlockIdx && curBlockIdx < SPLIT_COUNT * stage2CoreUsed - 1) {
                // stage2CoreUsed ~ 2 * stage2CoreUsed -1
                reduce_dgamma_dbeta_WSP2GM(divceil(divceil(N, SPLIT_COUNT), SPLIT_COUNT) * C,
                    (curBlockIdx % stage2CoreUsed) * castEleNum,
                    castEleNum,
                    (divceil(N, SPLIT_COUNT) - divceil(divceil(N, SPLIT_COUNT), SPLIT_COUNT)));
            } else if (curBlockIdx == stage2CoreUsed - 1) {
                // stage2CoreUsed -1
                reduce_dgamma_dbeta_WSP2GM(
                    0, curBlockIdx * castEleNum, tailCastNum, divceil(divceil(N, SPLIT_COUNT), SPLIT_COUNT));
            } else if (curBlockIdx == SPLIT_COUNT * stage2CoreUsed - 1) {
                // 2 * stage2CoreUsed -1
                reduce_dgamma_dbeta_WSP2GM(divceil(divceil(N, SPLIT_COUNT), SPLIT_COUNT) * C,
                    (curBlockIdx % stage2CoreUsed) * castEleNum,
                    tailCastNum,
                    (divceil(N, SPLIT_COUNT) - divceil(divceil(N, SPLIT_COUNT), SPLIT_COUNT)));
            }
        } else if constexpr (isDeterministic && !IsSameType<T, float>::value) {
#ifndef __CCE_KT_TEST__
            // wait core
            SyncAll();
#endif
            Initbuffer4stage2_mode2();
            // reduceSUM And Cast TO GM
            if (curBlockIdx < stage2CoreUsed - 1) {
                reduce_cast_dgamma_dbeta_WSP2GM(curBlockIdx * castEleNum, castEleNum, divceil(N, SPLIT_COUNT));
            } else if (curBlockIdx == stage2CoreUsed - 1) {
                reduce_cast_dgamma_dbeta_WSP2GM(curBlockIdx * castEleNum, tailCastNum, divceil(N, SPLIT_COUNT));
            }
        }
    }

private:
    // Pipe object
    TPipe pipe;
    // Que object
    TQue<QuePosition::VECIN, 1> in_queue_dy;
    TQue<QuePosition::VECIN, 1> in_queue_x;
    TQue<QuePosition::VECIN, 1> in_queue_gamma;
    TQue<QuePosition::VECIN, 1> in_queue_dgamma_channel;
    TQue<QuePosition::VECIN, 1> in_queue_dbeta_channel;
    TQue<QuePosition::VECIN, 1> cal_queue_dgamma_reduce;
    TQue<QuePosition::VECIN, 1> cal_queue_dbeta_reduce;
    TBuf<TPosition::VECCALC> in_tbuf_dy_T;
    TBuf<TPosition::VECCALC> in_tbuf_x_T;
    TBuf<TPosition::VECCALC> in_temp_x;
    TBuf<TPosition::VECCALC> in_temp_dy;
    TBuf<TPosition::VECCALC> in_tbuf_gamma_T;
    TBuf<TPosition::VECCALC> out_tbuf_dgamma_T;
    TBuf<TPosition::VECCALC> out_tbuf_dbeta_T;
    TBuf<TPosition::VECCALC> out_tbuf_dgamma_channel_T;
    TBuf<TPosition::VECCALC> out_tbuf_dbeta_channel_T;
    // Que for output
    TQue<QuePosition::VECOUT, 1> out_queue_dgamma;
    TQue<QuePosition::VECOUT, 1> out_queue_dbeta;
    TQue<QuePosition::VECOUT, 1> temp_queue_HXW;
    // Global Memory
    GlobalTensor<T> dy_gm;
    GlobalTensor<T> mean_gm;
    GlobalTensor<T> rstd_gm;
    GlobalTensor<T> x_gm;
    GlobalTensor<T> gamma_gm;
    GlobalTensor<T> beta_gm;
    GlobalTensor<float> dgamma_workspace;
    GlobalTensor<float> dbeta_workspace;
    GlobalTensor<T> dx_gm;
    GlobalTensor<T> dgamma_gm;
    GlobalTensor<T> dbeta_gm;
    uint32_t Tiling_key;
    bool isGammaEqual = true;
    uint32_t N;
    uint32_t C;
    uint32_t G;
    uint32_t HXW;
    uint32_t NXG;
    uint32_t C_G;
    uint32_t all_ele_num;
    // number of calculations on each core
    uint32_t ele_num_per_group;
    // number of tiles on each core
    uint32_t ele_num_per_channel;
    uint32_t task_idx;
    // number of calculations in each tile
    uint32_t tileLength;
    uint32_t task_num_per_core;
    uint32_t task_num_per_tail_core;
    uint32_t tail_core;
    uint32_t cur_core_task_num;
    uint32_t workSpaceSize;
    uint32_t stage2CoreUsed;
    uint32_t castEleNum;
    uint32_t tailCastNum;
    int32_t curBlockIdx;
    int32_t start_task_id;
    float meanScalar;
    float rstdScalar;
    uint32_t mode2_ub_capacity_ele;
    uint32_t mode2_ub_iteration_num;
    uint32_t mode2_ub_tail_num;
    uint32_t coreBatchParts = 0;
    bool dxIsRequire;
    bool dgammaIsRequire;
    bool dbetaIsRequire;
    uint32_t T_per_block;
    const uint32_t float_per_block = 8;
    const uint16_t DST_BLK_STRIDE = 1;
    const uint16_t SRC_BLK_STRIDE = 1;
    const uint8_t DST_REP_STRIDE = 8;
    const uint8_t SRC_REP_STRIDE = 8;
    const uint32_t BLOCK_BYTES = 32;
    const uint32_t MAX_REPEAT_TIMES = 255;
    const uint32_t COM_REPEAT_TIMES = 64;
    const int32_t MODE_0 = 0;
    const int32_t MODE_1 = 1;
    const int32_t MODE_2 = 2;
    const int32_t MODE_3 = 3;
    const int32_t MODE_4 = 4;
    const int32_t MODE_5 = 5;
    const int doubleBuffer = 2;
    const int32_t VECTOR_ONCE_BYTES = 256;

private:
    //----------------------------------------------MODE0--------------------------------------//
    __aicore__ inline void CopyIn_Mode_0(int32_t task_idx)
    {
        // alloc tensor from queue memory
        uint32_t offset = task_idx * this->ele_num_per_group;
        uint32_t once_process_ele_num = this->ele_num_per_group;
        LocalTensor<float> dy_local = in_queue_dy.AllocTensor<float>();
        Custom_DataCopy_In(dy_local, in_tbuf_dy_T, dy_gm, offset, once_process_ele_num);
        in_queue_dy.EnQue(dy_local);
        LocalTensor<float> x_local = in_queue_x.AllocTensor<float>();
        Custom_DataCopy_In(x_local, in_tbuf_x_T, x_gm, offset, once_process_ele_num);
        event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        Get_Mean_Rstd(task_idx);
        event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIDSToV);
        WaitFlag<HardEvent::S_V>(eventIDSToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        Adds_Muls_template(x_local,
            x_local,
            (float(-1.0) * this->meanScalar * this->rstdScalar),
            this->rstdScalar,
            this->ele_num_per_group);
        in_queue_x.EnQue(x_local);
    }

    __aicore__ inline void Compute_Mode_0(int32_t task_idx)
    {
        // deque input tensors from VECIN queue
        LocalTensor<float> dy_local = in_queue_dy.DeQue<float>();
        LocalTensor<float> x_local = in_queue_x.DeQue<float>();
        LocalTensor<float> temp_1_local = out_queue_dbeta.AllocTensor<float>();
        LocalTensor<float> temp_2_local = out_queue_dgamma.AllocTensor<float>();
        LocalTensor<float> temp_hxw_local = temp_queue_HXW.AllocTensor<float>();

        for (int32_t C_G_idx = 0; C_G_idx < this->C_G; C_G_idx++) {
            int32_t is_align = (C_G_idx * this->ele_num_per_channel) % this->float_per_block;
            if (is_align == 0) {
                float temp1I = ReduceSum_template(temp_hxw_local,
                    dy_local[C_G_idx * this->ele_num_per_channel],
                    temp_hxw_local,
                    this->ele_num_per_channel,
                    true);
                temp_1_local.SetValue(C_G_idx, temp1I);
                event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
                SetFlag<HardEvent::S_V>(eventIDSToV);
                WaitFlag<HardEvent::S_V>(eventIDSToV);
                Mul(temp_hxw_local,
                    x_local[C_G_idx * this->ele_num_per_channel],
                    dy_local[C_G_idx * this->ele_num_per_channel],
                    this->ele_num_per_channel);
                float temp2I =
                    ReduceSum_template(temp_hxw_local, temp_hxw_local, temp_hxw_local, this->ele_num_per_channel, true);
                temp_2_local.SetValue(C_G_idx, temp2I);
            } else {
                uint32_t gm_offset = task_idx * this->ele_num_per_group + C_G_idx * this->ele_num_per_channel;
                LocalTensor<float> unalign_channel_x = in_temp_x.Get<float>(this->ele_num_per_channel);
                LocalTensor<float> unalign_channel_dy = in_temp_dy.Get<float>(this->ele_num_per_channel);
                Custom_DataCopy_In(unalign_channel_x, in_tbuf_x_T, x_gm, gm_offset, this->ele_num_per_channel);
                Custom_DataCopy_In(unalign_channel_dy, in_tbuf_dy_T, dy_gm, gm_offset, this->ele_num_per_channel);
                event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
                WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
                Adds_Muls_template(unalign_channel_x,
                    unalign_channel_x,
                    (float(-1.0) * this->meanScalar * this->rstdScalar),
                    this->rstdScalar,
                    this->ele_num_per_channel);
                float temp1I = ReduceSum_template(
                    temp_hxw_local, unalign_channel_dy, temp_hxw_local, this->ele_num_per_channel, true);
                Mul(temp_hxw_local, unalign_channel_x, unalign_channel_dy, this->ele_num_per_channel);
                float temp2I =
                    ReduceSum_template(temp_hxw_local, temp_hxw_local, temp_hxw_local, this->ele_num_per_channel, true);
                PipeBarrier<PIPE_ALL>();
                temp_1_local.SetValue(C_G_idx, temp1I);
                temp_2_local.SetValue(C_G_idx, temp2I);
            }
        }
        temp_queue_HXW.FreeTensor(temp_hxw_local);
        Mode_Selection(task_idx, temp_2_local, temp_1_local);
        if (dxIsRequire) {
            uint32_t channel_idx = (task_idx % this->G) * this->C_G;
            LocalTensor<float> gamma_local = in_queue_gamma.AllocTensor<float>();
            Custom_DataCopy_In(gamma_local, in_tbuf_gamma_T, gamma_gm, channel_idx, this->C_G);
            event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            float gamma = 0;
            float c1 = 0;
            float c2 = 0;
            Mul_ReduceSum_template(temp_1_local, temp_2_local, gamma_local, C_G, c1, c2);
            for (int32_t C_G_idx = 0; C_G_idx < this->C_G; C_G_idx++) {
                int32_t is_align = (C_G_idx * this->ele_num_per_channel) % this->T_per_block;
                uint32_t gm_offset = task_idx * this->ele_num_per_group + C_G_idx * this->ele_num_per_channel;
                gamma = gamma_local.GetValue(C_G_idx);
                event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
                SetFlag<HardEvent::S_V>(eventIDSToV);
                if (is_align == 0) {
                    uint32_t offset = C_G_idx * this->ele_num_per_channel;
                    Adds_Muls_template(x_local[offset],
                        x_local[offset],
                        (this->rstdScalar * c1),
                        (this->rstdScalar * c2),
                        this->ele_num_per_channel);
                    WaitFlag<HardEvent::S_V>(eventIDSToV);
                    Muls(dy_local[offset], dy_local[offset], (this->rstdScalar * gamma), this->ele_num_per_channel);
                    LocalTensor<float> temp_hxw_local = temp_queue_HXW.AllocTensor<float>();
                    Sub(temp_hxw_local, dy_local[offset], x_local[offset], this->ele_num_per_channel);
                    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                    Custom_DataCopy_Out(temp_hxw_local, gm_offset, in_tbuf_dy_T, ele_num_per_channel);
                    temp_queue_HXW.FreeTensor(temp_hxw_local);
                } else {
                    uint32_t offset = floor(C_G_idx * this->ele_num_per_channel, this->T_per_block);
                    PipeBarrier<PIPE_ALL>();
                    Custom_DataCopy_In(x_local, offset, in_tbuf_x_T, x_gm, gm_offset, this->ele_num_per_channel);
                    Custom_DataCopy_In(dy_local, offset, in_tbuf_dy_T, dy_gm, gm_offset, this->ele_num_per_channel);
                    event_t eventIDMTE2ToV0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
                    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
                    Adds_Muls_template(x_local[offset],
                        x_local[offset],
                        (this->rstdScalar * c1 - this->meanScalar * this->rstdScalar * this->rstdScalar * c2),
                        (this->rstdScalar * this->rstdScalar * c2),
                        this->ele_num_per_channel);
                    WaitFlag<HardEvent::S_V>(eventIDSToV);
                    Muls(dy_local[offset], dy_local[offset], (this->rstdScalar * gamma), this->ele_num_per_channel);
                    LocalTensor<float> temp_hxw_local = temp_queue_HXW.AllocTensor<float>();
                    Sub(temp_hxw_local, dy_local[offset], x_local[offset], this->ele_num_per_channel);
                    PipeBarrier<PIPE_ALL>();
                    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                    Custom_DataCopy_Out(temp_hxw_local, gm_offset, in_tbuf_dy_T, ele_num_per_channel);
                    temp_queue_HXW.FreeTensor(temp_hxw_local);
                }
            }
            in_queue_gamma.FreeTensor(gamma_local);
        }
        out_queue_dbeta.FreeTensor(temp_1_local);
        out_queue_dgamma.FreeTensor(temp_2_local);
        in_queue_dy.FreeTensor(dy_local);
        in_queue_x.FreeTensor(x_local);
    }
    //----------------------------------------------MODE1--------------------------------------//
    __aicore__ inline void Compute_Mode_1(int32_t task_idx)
    {
        LocalTensor<float> temp_1_local = out_queue_dbeta.AllocTensor<float>();
        LocalTensor<float> temp_2_local = out_queue_dgamma.AllocTensor<float>();
        uint32_t base_offset = task_idx * this->ele_num_per_group;
        uint32_t once_process_ele_num = this->ele_num_per_channel;
        LocalTensor<T> dy_T_local;
        LocalTensor<T> x_T_local;
        LocalTensor<float> dy_local;
        LocalTensor<float> x_local;
        LocalTensor<float> temp_hxw_local;
        for (int32_t C_G_idx = 0; C_G_idx < this->C_G; C_G_idx++) {
            uint32_t offset = base_offset + C_G_idx * this->ele_num_per_channel;
            dy_local = in_queue_dy.AllocTensor<float>();
            Custom_DataCopy_In(dy_local, in_tbuf_dy_T, dy_gm, offset, once_process_ele_num);
            x_local = in_queue_x.AllocTensor<float>();
            Custom_DataCopy_In(x_local, in_tbuf_x_T, x_gm, offset, once_process_ele_num);
            event_t eventIDMTE2ToV0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
            temp_hxw_local = temp_queue_HXW.AllocTensor<float>();
            float temp1I =
                ReduceSum_template(temp_hxw_local, dy_local, temp_hxw_local, this->ele_num_per_channel, true);
            Adds_Muls_template(x_local,
                x_local,
                (-1.0f * this->meanScalar * this->rstdScalar),
                (this->rstdScalar),
                this->ele_num_per_channel);
            Mul(temp_hxw_local, x_local, dy_local, this->ele_num_per_channel);
            in_queue_x.FreeTensor(x_local);
            in_queue_dy.FreeTensor(dy_local);
            float temp2I =
                ReduceSum_template(temp_hxw_local, temp_hxw_local, temp_hxw_local, this->ele_num_per_channel, true);
            temp_queue_HXW.FreeTensor(temp_hxw_local);
            temp_1_local.SetValue(C_G_idx, temp1I);
            temp_2_local.SetValue(C_G_idx, temp2I);
        }
        Mode_Selection(task_idx, temp_2_local, temp_1_local);
        if (dxIsRequire) {
            float c1 = 0;
            float c2 = 0;
            float gamma = 0;
            uint32_t channel_idx = (task_idx % this->G) * this->C_G;
            LocalTensor<float> gamma_local = in_queue_gamma.AllocTensor<float>();
            Custom_DataCopy_In(gamma_local, in_tbuf_gamma_T, gamma_gm, channel_idx, this->C_G);
            event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            Mul_ReduceSum_template(temp_1_local, temp_2_local, gamma_local, C_G, c1, c2);
            out_queue_dbeta.FreeTensor(temp_1_local);
            out_queue_dgamma.FreeTensor(temp_2_local);
            for (int32_t C_G_idx = 0; C_G_idx < this->C_G; C_G_idx++) {
                uint32_t offset = base_offset + C_G_idx * this->ele_num_per_channel;
                gamma = (float)gamma_local.GetValue(C_G_idx);
                event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
                SetFlag<HardEvent::S_V>(eventIDSToV);
                x_local = in_queue_x.AllocTensor<float>();
                Custom_DataCopy_In(x_local, in_tbuf_x_T, x_gm, offset, once_process_ele_num);
                dy_local = in_queue_dy.AllocTensor<float>();
                Custom_DataCopy_In(dy_local, in_tbuf_dy_T, dy_gm, offset, once_process_ele_num);
                event_t eventIDMTE2ToV0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
                WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
                WaitFlag<HardEvent::S_V>(eventIDSToV);
                Muls(dy_local, dy_local, gamma, this->ele_num_per_channel);
                Adds_Muls_template(x_local,
                    x_local,
                    (-1.0f * (this->meanScalar * this->rstdScalar * c2 - c1)),
                    (this->rstdScalar * c2),
                    this->ele_num_per_channel);
                Sub(dy_local, dy_local, x_local, this->ele_num_per_channel);
                in_queue_x.FreeTensor(x_local);
                temp_hxw_local = temp_queue_HXW.AllocTensor<float>();
                Muls(temp_hxw_local, dy_local, (this->rstdScalar), this->ele_num_per_channel);
                in_queue_dy.FreeTensor(dy_local);
                event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                Custom_DataCopy_Out(temp_hxw_local, offset, in_tbuf_dy_T, ele_num_per_channel);
                temp_queue_HXW.FreeTensor(temp_hxw_local);
            }
            in_queue_gamma.FreeTensor(gamma_local);
        } else {
            out_queue_dbeta.FreeTensor(temp_1_local);
            out_queue_dgamma.FreeTensor(temp_2_local);
        }
    }
    //----------------------------------------------MODE2--------------------------------------//
    __aicore__ inline void Compute_Mode_2(int32_t task_idx)
    {
        // deque input tensors from VECIN queue
        LocalTensor<float> x_local[2] = {in_queue_x.AllocTensor<float>(), in_queue_x.AllocTensor<float>()};
        LocalTensor<float> dy_local[2] = {in_queue_dy.AllocTensor<float>(), in_queue_dy.AllocTensor<float>()};
        LocalTensor<float> temp_hxw_local[2] = {
            temp_queue_HXW.AllocTensor<float>(), temp_queue_HXW.AllocTensor<float>()};
        LocalTensor<T> dy_T_local[2];
        LocalTensor<T> x_T_local[2];
        const int DOUBLE_BUFFER_ = 2;
        int restChannelStartIdx = C_G / DOUBLE_BUFFER_ * DOUBLE_BUFFER_;
        uint32_t channel_idx = (task_idx % this->G) * this->C_G;
        LocalTensor<float> temp_1_local = out_queue_dbeta.AllocTensor<float>();
        LocalTensor<float> temp_2_local = out_queue_dgamma.AllocTensor<float>();
        uint32_t offset[2];
        float temp1I[2];
        float temp2I[2];

        uint32_t base_offset = task_idx * this->ele_num_per_group;
        uint32_t once_process_ele_num = this->ele_num_per_channel;
        for (int32_t C_G_idx = 0; C_G_idx < this->C_G / DOUBLE_BUFFER_; C_G_idx++) {
            for (int i = 0; i < DOUBLE_BUFFER_; i++) {
                offset[i] = base_offset + (C_G_idx * DOUBLE_BUFFER_ + i) * this->ele_num_per_channel;
            }
            for (int i = 0; i < DOUBLE_BUFFER_; i++) {
                Custom_DataCopy_In(dy_local[i], in_tbuf_dy_T, dy_gm, offset[i], once_process_ele_num);
                Custom_DataCopy_In(x_local[i], in_tbuf_x_T, x_gm, offset[i], once_process_ele_num);
            }
            event_t eventIDMTE2ToV0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
            for (int i = 0; i < DOUBLE_BUFFER_; i++) {
                temp1I[i] = ReduceSum_template(
                    temp_hxw_local[i], dy_local[i], temp_hxw_local[i], this->ele_num_per_channel, true);
            }
            for (int i = 0; i < DOUBLE_BUFFER_; i++) {
                Adds_Muls_template(x_local[i],
                    x_local[i],
                    (-1.0f * this->meanScalar * this->rstdScalar),
                    this->rstdScalar,
                    this->ele_num_per_channel);
            }
            for (int i = 0; i < DOUBLE_BUFFER_; i++) {
                Mul(temp_hxw_local[i], x_local[i], dy_local[i], this->ele_num_per_channel);
            }
            for (int i = 0; i < DOUBLE_BUFFER_; i++) {
                temp2I[i] = ReduceSum_template(
                    temp_hxw_local[i], temp_hxw_local[i], temp_hxw_local[i], this->ele_num_per_channel, true);
            }
            for (int i = 0; i < DOUBLE_BUFFER_; i++) {
                temp_1_local.SetValue(C_G_idx * DOUBLE_BUFFER_ + i, temp1I[i]);
                temp_2_local.SetValue(C_G_idx * DOUBLE_BUFFER_ + i, temp2I[i]);
            }
        }
        if (C_G != restChannelStartIdx) {
            offset[0] = base_offset + restChannelStartIdx * this->ele_num_per_channel;
            Custom_DataCopy_In(dy_local[0], in_tbuf_dy_T, dy_gm, offset[0], once_process_ele_num);
            Custom_DataCopy_In(x_local[0], in_tbuf_x_T, x_gm, offset[0], once_process_ele_num);
            event_t eventIDMTE2ToV0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
            temp1I[0] =
                ReduceSum_template(temp_hxw_local[0], dy_local[0], temp_hxw_local[0], this->ele_num_per_channel, true);
            Adds_Muls_template(x_local[0],
                x_local[0],
                (-1.0f * this->meanScalar * this->rstdScalar),
                this->rstdScalar,
                this->ele_num_per_channel);
            Mul(temp_hxw_local[0], x_local[0], dy_local[0], this->ele_num_per_channel);
            temp2I[0] = ReduceSum_template(
                temp_hxw_local[0], temp_hxw_local[0], temp_hxw_local[0], this->ele_num_per_channel, true);
            temp_2_local.SetValue(restChannelStartIdx, temp2I[0]);
            temp_1_local.SetValue(restChannelStartIdx, temp1I[0]);
        }
        temp_queue_HXW.FreeTensor(temp_hxw_local[0]);
        temp_queue_HXW.FreeTensor(temp_hxw_local[1]);
        in_queue_dy.FreeTensor(dy_local[0]);
        in_queue_dy.FreeTensor(dy_local[1]);
        in_queue_x.FreeTensor(x_local[0]);
        in_queue_x.FreeTensor(x_local[1]);
        Mode_Selection(task_idx, temp_2_local, temp_1_local);
        if (dxIsRequire) {
            float gamma[2] = {0, 0};
            float c1 = 0;
            float c2 = 0;
            LocalTensor<float> gamma_local = in_queue_gamma.AllocTensor<float>();
            Custom_DataCopy_In(gamma_local, in_tbuf_gamma_T, gamma_gm, channel_idx, this->C_G);
            event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            Mul_ReduceSum_template(temp_1_local, temp_2_local, gamma_local, C_G, c1, c2);
            out_queue_dgamma.FreeTensor(temp_2_local);
            out_queue_dbeta.FreeTensor(temp_1_local);
            DataCopyParams copyParams{1, (uint16_t)(this->ele_num_per_channel * sizeof(float)), 0, 0};
            for (int32_t C_G_idx = 0; C_G_idx < this->C_G / DOUBLE_BUFFER_; C_G_idx++) {
                for (int i = 0; i < DOUBLE_BUFFER_; i++) {
                    offset[i] = base_offset + (C_G_idx * DOUBLE_BUFFER_ + i) * this->ele_num_per_channel;
                    gamma[i] = (float)gamma_local.GetValue(C_G_idx * DOUBLE_BUFFER_ + i);
                }
                event_t eventIDSToV0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
                SetFlag<HardEvent::S_V>(eventIDSToV0);
                for (int i = 0; i < DOUBLE_BUFFER_; i++) {
                    x_local[i] = in_queue_x.AllocTensor<float>();
                    dy_local[i] = in_queue_dy.AllocTensor<float>();
                    Custom_DataCopy_In(x_local[i], in_tbuf_x_T, x_gm, offset[i], once_process_ele_num);
                    Custom_DataCopy_In(dy_local[i], in_tbuf_dy_T, dy_gm, offset[i], once_process_ele_num);
                }
                event_t eventIDMTE2ToV0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
                WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
                WaitFlag<HardEvent::S_V>(eventIDSToV0);
                for (int i = 0; i < DOUBLE_BUFFER_; i++) {
                    Muls(dy_local[i], dy_local[i], gamma[i], this->ele_num_per_channel);
                    Adds_Muls_template(x_local[i],
                        x_local[i],
                        (-1.0f * (this->meanScalar * this->rstdScalar * c2 - c1)),
                        (this->rstdScalar * c2),
                        this->ele_num_per_channel);
                    Sub(dy_local[i], dy_local[i], x_local[i], this->ele_num_per_channel);
                    in_queue_x.FreeTensor(x_local[i]);
                    temp_hxw_local[i] = temp_queue_HXW.AllocTensor<float>();
                    Muls(temp_hxw_local[i], dy_local[i], this->rstdScalar, this->ele_num_per_channel);
                    in_queue_dy.FreeTensor(dy_local[i]);
                    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                    Custom_DataCopy_Out(temp_hxw_local[i], offset[i], in_tbuf_dy_T, ele_num_per_channel);
                    temp_queue_HXW.FreeTensor(temp_hxw_local[i]);
                }
            }
            if (C_G != restChannelStartIdx) {
                offset[0] = base_offset + restChannelStartIdx * this->ele_num_per_channel;
                gamma[0] = (float)gamma_local.GetValue(restChannelStartIdx);
                event_t eventIDSToV0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
                SetFlag<HardEvent::S_V>(eventIDSToV0);
                x_local[0] = in_queue_x.AllocTensor<float>();
                dy_local[0] = in_queue_dy.AllocTensor<float>();
                Custom_DataCopy_In(x_local[0], in_tbuf_x_T, x_gm, offset[0], once_process_ele_num);
                Custom_DataCopy_In(dy_local[0], in_tbuf_dy_T, dy_gm, offset[0], once_process_ele_num);
                event_t eventIDMTE2ToV0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
                WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
                WaitFlag<HardEvent::S_V>(eventIDSToV0);
                Muls(dy_local[0], dy_local[0], gamma[0], this->ele_num_per_channel);
                Adds_Muls_template(x_local[0],
                    x_local[0],
                    (-1.0f * (this->meanScalar * this->rstdScalar * c2 - c1)),
                    (this->rstdScalar * c2),
                    this->ele_num_per_channel);
                Sub(dy_local[0], dy_local[0], x_local[0], this->ele_num_per_channel);
                in_queue_x.FreeTensor(x_local[0]);
                temp_hxw_local[0] = temp_queue_HXW.AllocTensor<float>();
                Muls(temp_hxw_local[0], dy_local[0], this->rstdScalar, this->ele_num_per_channel);
                in_queue_dy.FreeTensor(dy_local[0]);
                event_t eventIDVToMTE30 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                SetFlag<HardEvent::V_MTE3>(eventIDVToMTE30);
                WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE30);
                Custom_DataCopy_Out(temp_hxw_local[0], offset[0], in_tbuf_dy_T, ele_num_per_channel);
                temp_queue_HXW.FreeTensor(temp_hxw_local[0]);
            }
            in_queue_gamma.FreeTensor(gamma_local);
        } else {
            out_queue_dgamma.FreeTensor(temp_2_local);
            out_queue_dbeta.FreeTensor(temp_1_local);
        }
    }
    //----------------------------------------------MODE3--------------------------------------//
    __aicore__ inline void CopyIn_x(float &corr1, float &temp1I, uint32_t offset, uint32_t process_num)
    {
        LocalTensor<float> x_local = in_queue_x.AllocTensor<float>();
        LocalTensor<float> dy_local = in_queue_dy.AllocTensor<float>();
        Custom_DataCopy_In(dy_local, in_tbuf_dy_T, dy_gm, offset, process_num);
        Custom_DataCopy_In(x_local, in_tbuf_x_T, x_gm, offset, process_num);
        in_queue_x.EnQue(x_local);

        event_t eventIDMTE2ToV0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
        LocalTensor<float> temp_hxw_local = temp_queue_HXW.AllocTensor<float>();
        float adjustedSum = ReduceSum_template(temp_hxw_local, dy_local, temp_hxw_local, process_num, true) - corr1;
        float tempSum = adjustedSum + temp1I;
        corr1 = (tempSum - temp1I) - adjustedSum;
        temp1I = tempSum;
        temp_queue_HXW.FreeTensor(temp_hxw_local);
        in_queue_dy.EnQue(dy_local);
    }

    __aicore__ inline void compute_dgamma(uint32_t offset, uint32_t process_num)
    {
        LocalTensor<float> x_local = in_queue_x.DeQue<float>();
        Adds_Muls_template(x_local,
            x_local,
            (-1.0f * this->meanScalar * this->rstdScalar),
            (this->rstdScalar),
            this->mode2_ub_capacity_ele);
        LocalTensor<float> temp_hxw_local = temp_queue_HXW.AllocTensor<float>();
        LocalTensor<float> dy_local = in_queue_dy.DeQue<float>();
        Mul(temp_hxw_local, x_local, dy_local, process_num);
        temp_queue_HXW.EnQue(temp_hxw_local);
        in_queue_dy.FreeTensor(dy_local);
        in_queue_x.FreeTensor(x_local);
    }

    __aicore__ inline void Copyout_dgamma(float &corr2, float &temp2I, uint32_t offset, uint32_t process_num)
    {
        LocalTensor<float> temp_hxw_local = temp_queue_HXW.DeQue<float>();
        float adjustedSum =
            ReduceSum_template(temp_hxw_local, temp_hxw_local, temp_hxw_local, process_num, true) - corr2;
        float tempSum = adjustedSum + temp2I;
        corr2 = (tempSum - temp2I) - adjustedSum;
        temp2I = tempSum;
        temp_queue_HXW.FreeTensor(temp_hxw_local);
    }

    __aicore__ inline void Compute_Mode_3(int32_t task_idx)
    {
        // deque input tensors from VECIN queue
        LocalTensor<float> temp_1_local = out_queue_dbeta.AllocTensor<float>();
        LocalTensor<float> temp_2_local = out_queue_dgamma.AllocTensor<float>();
        float temp1I = 0;
        float temp2I = 0;
        float corr1 = 0;
        float corr2 = 0;
        for (int32_t iter_C_G_idx = 0; iter_C_G_idx < this->C_G * this->mode2_ub_iteration_num; iter_C_G_idx++) {
            uint32_t offset = task_idx * this->ele_num_per_group +
                              iter_C_G_idx / this->mode2_ub_iteration_num * this->ele_num_per_channel +
                              iter_C_G_idx % this->mode2_ub_iteration_num * this->mode2_ub_capacity_ele;
            if (iter_C_G_idx % this->mode2_ub_iteration_num != this->mode2_ub_iteration_num - 1) {
                CopyIn_x(corr1, temp1I, offset, this->mode2_ub_capacity_ele);
                compute_dgamma(offset, this->mode2_ub_capacity_ele);
                Copyout_dgamma(corr2, temp2I, offset, this->mode2_ub_capacity_ele);
            } else {
                CopyIn_x(corr1, temp1I, offset, this->mode2_ub_tail_num);
                compute_dgamma(offset, this->mode2_ub_tail_num);
                Copyout_dgamma(corr2, temp2I, offset, this->mode2_ub_tail_num);
                temp_1_local.SetValue((int)(iter_C_G_idx / this->mode2_ub_iteration_num), temp1I);
                temp_2_local.SetValue((int)(iter_C_G_idx / this->mode2_ub_iteration_num), temp2I);
                corr1 = 0;
                corr2 = 0;
                temp1I = 0;
                temp2I = 0;
            }
        }
        Mode_Selection(task_idx, temp_2_local, temp_1_local);
        if (dxIsRequire) {
            float c1 = 0;
            float c2 = 0;
            uint32_t channel_idx = (task_idx % this->G) * this->C_G;
            LocalTensor<float> gamma_local = in_queue_gamma.AllocTensor<float>();
            Custom_DataCopy_In(gamma_local, in_tbuf_gamma_T, gamma_gm, channel_idx, this->C_G);
            event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            Mul_ReduceSum_template(temp_1_local, temp_2_local, gamma_local, C_G, c1, c2);
            out_queue_dbeta.FreeTensor(temp_1_local);
            out_queue_dgamma.FreeTensor(temp_2_local);
            float gamma = 0;
            for (int32_t iter_C_G_idx = 0; iter_C_G_idx < this->C_G * this->mode2_ub_iteration_num; iter_C_G_idx++) {
                uint32_t offset = task_idx * this->ele_num_per_group +
                                  iter_C_G_idx / this->mode2_ub_iteration_num * this->ele_num_per_channel +
                                  iter_C_G_idx % this->mode2_ub_iteration_num * this->mode2_ub_capacity_ele;
                gamma = (float)gamma_local.GetValue(iter_C_G_idx / this->mode2_ub_iteration_num);
                event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
                SetFlag<HardEvent::S_V>(eventIDSToV);
                if (iter_C_G_idx % this->mode2_ub_iteration_num != this->mode2_ub_iteration_num - 1) {
                    LocalTensor<float> x_local = in_queue_x.AllocTensor<float>();
                    LocalTensor<float> dy_local = in_queue_dy.AllocTensor<float>();
                    Custom_DataCopy_In(x_local, in_tbuf_x_T, x_gm, offset, this->mode2_ub_capacity_ele);
                    Custom_DataCopy_In(dy_local, in_tbuf_dy_T, dy_gm, offset, this->mode2_ub_capacity_ele);
                    event_t eventIDMTE2ToV0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
                    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
                    WaitFlag<HardEvent::S_V>(eventIDSToV);
                    Muls(dy_local, dy_local, gamma, this->mode2_ub_capacity_ele);
                    Adds_Muls_template(x_local,
                        x_local,
                        (-1.0f * (this->meanScalar * this->rstdScalar * c2 - c1)),
                        (this->rstdScalar * c2),
                        this->mode2_ub_capacity_ele);
                    Sub(dy_local, dy_local, x_local, this->mode2_ub_capacity_ele);
                    LocalTensor<float> temp_hxw_local = temp_queue_HXW.AllocTensor<float>();
                    Muls(temp_hxw_local, dy_local, this->rstdScalar, this->mode2_ub_capacity_ele);
                    in_queue_x.FreeTensor(x_local);
                    in_queue_dy.FreeTensor(dy_local);
                    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                    Custom_DataCopy_Out(temp_hxw_local, offset, in_tbuf_dy_T, mode2_ub_capacity_ele);
                    temp_queue_HXW.FreeTensor(temp_hxw_local);
                } else {
                    LocalTensor<float> x_local = in_queue_x.AllocTensor<float>();
                    LocalTensor<float> dy_local = in_queue_dy.AllocTensor<float>();
                    Custom_DataCopy_In(x_local, in_tbuf_x_T, x_gm, offset, this->mode2_ub_tail_num);
                    Custom_DataCopy_In(dy_local, in_tbuf_dy_T, dy_gm, offset, this->mode2_ub_tail_num);
                    event_t eventIDMTE2ToV0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
                    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
                    WaitFlag<HardEvent::S_V>(eventIDSToV);
                    Muls(dy_local, dy_local, gamma, this->mode2_ub_tail_num);
                    Adds_Muls_template(x_local,
                        x_local,
                        (-1.0f * (this->meanScalar * this->rstdScalar * c2 - c1)),
                        (this->rstdScalar * c2),
                        this->mode2_ub_tail_num);
                    Sub(dy_local, dy_local, x_local, this->mode2_ub_tail_num);
                    LocalTensor<float> temp_hxw_local = temp_queue_HXW.AllocTensor<float>();
                    Muls(temp_hxw_local, dy_local, this->rstdScalar, this->mode2_ub_tail_num);
                    in_queue_x.FreeTensor(x_local);
                    in_queue_dy.FreeTensor(dy_local);
                    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                    Custom_DataCopy_Out(temp_hxw_local, offset, in_tbuf_dy_T, mode2_ub_tail_num);
                    temp_queue_HXW.FreeTensor(temp_hxw_local);
                }
            }
            in_queue_gamma.FreeTensor(gamma_local);
        } else {
            out_queue_dbeta.FreeTensor(temp_1_local);
            out_queue_dgamma.FreeTensor(temp_2_local);
        }
    }
    //----------------------------------------------MODE5--------------------------------------//
    __aicore__ inline void Compute_Mode_5(int32_t task_idx)
    {
        // deque input tensors from VECIN queue
        LocalTensor<float> dy_local = in_queue_dy.DeQue<float>();
        LocalTensor<float> x_local = in_queue_x.DeQue<float>();
        LocalTensor<float> temp_1_local = out_queue_dbeta.AllocTensor<float>();
        LocalTensor<float> temp_2_local = out_queue_dgamma.AllocTensor<float>();
        DataCopy(temp_1_local, dy_local, ceil(this->C_G, this->float_per_block));
        PipeBarrier<PIPE_ALL>();
        Mul(temp_2_local, x_local, dy_local, this->C_G);
        Mode_Selection(task_idx, temp_2_local, temp_1_local);
        if (dxIsRequire) {
            uint32_t channel_idx = (task_idx % this->G) * this->C_G;
            LocalTensor<float> gamma_local = in_queue_gamma.AllocTensor<float>();
            Custom_DataCopy_In(gamma_local, in_tbuf_gamma_T, gamma_gm, channel_idx, this->C_G);
            event_t eventIDMTE2ToV1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV1);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV1);
            Mul(dy_local, dy_local, gamma_local, this->C_G);
            float sumC1 = 0;
            float sumC2 = 0;
            Mul_ReduceSum_template(temp_1_local, temp_2_local, gamma_local, C_G, sumC1, sumC2);
            out_queue_dbeta.FreeTensor(temp_1_local);
            out_queue_dgamma.FreeTensor(temp_2_local);
            in_queue_gamma.FreeTensor(gamma_local);
            Adds_Muls_template(x_local, x_local, this->rstdScalar * sumC1, this->rstdScalar * sumC2, this->C_G);
            Muls(dy_local, dy_local, this->rstdScalar, this->C_G);
            LocalTensor<float> temp_hxw_local = temp_queue_HXW.AllocTensor<float>();
            Sub(temp_hxw_local, dy_local, x_local, this->C_G);
            in_queue_dy.FreeTensor(dy_local);
            in_queue_x.FreeTensor(x_local);
            uint32_t gm_offset = task_idx * this->C_G;
            event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            Custom_DataCopy_Out(temp_hxw_local, gm_offset, in_tbuf_dy_T, this->C_G);
            event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            temp_queue_HXW.FreeTensor(temp_hxw_local);
        } else {
            out_queue_dbeta.FreeTensor(temp_1_local);
            out_queue_dgamma.FreeTensor(temp_2_local);
            in_queue_dy.FreeTensor(dy_local);
            in_queue_x.FreeTensor(x_local);
        }
    }
    //----------------------------------------------tools_func--------------------------------------//
    __aicore__ inline void Init_Output_GM()
    {
        if constexpr (!isDeterministic && IsSameType<T, float>::value) {
            if (this->curBlockIdx == 0) {
                InitOutput<float>(dgamma_gm, C, 0.0f);
                InitOutput<float>(dbeta_gm, C, 0.0f);
            }
        } else if constexpr (!isDeterministic && !IsSameType<T, float>::value) {
            if (this->curBlockIdx == 0) {
                InitOutput<float>(dgamma_workspace, C, 0.0f);
                InitOutput<float>(dbeta_workspace, C, 0.0f);
            }
        } else if constexpr (isDeterministic && IsSameType<T, float>::value) {
            if (this->curBlockIdx == 0) {
                InitOutput<float>(dgamma_workspace, workSpaceSize, 0.0f);
                InitOutput<float>(dbeta_workspace, workSpaceSize, 0.0f);
                InitOutput<float>(dgamma_gm, C, 0.0f);
                InitOutput<float>(dbeta_gm, C, 0.0f);
            }
        } else if constexpr (isDeterministic && !IsSameType<T, float>::value) {
            if (this->curBlockIdx == 0) {
                InitOutput<float>(dgamma_workspace, workSpaceSize, 0.0f);
                InitOutput<float>(dbeta_workspace, workSpaceSize, 0.0f);
            }
        }
    }

    __aicore__ inline void Init_Unified_Buffer(GroupNormGradTilingData *tiling_data)
    {
        if constexpr (IsSameType<T, float>::value) {
            if (Tiling_key == MODE_0) {
                // VEC_IN
                uint32_t HxW_alloc_space = ceil(this->ele_num_per_channel, T_per_block) * sizeof(T);
                uint32_t C_GxHxW_alloc_space = ceil(this->ele_num_per_group, T_per_block) * sizeof(T);
                uint32_t NxG_alloc_space = T_per_block * sizeof(T);
                uint32_t C_G_alloc_space = ceil(this->C_G, T_per_block) * sizeof(T);
                pipe.InitBuffer(in_queue_dy, 1, C_GxHxW_alloc_space);
                pipe.InitBuffer(in_queue_x, 1, C_GxHxW_alloc_space);
                pipe.InitBuffer(in_queue_gamma, 1, C_G_alloc_space);
                // VEC_OUT
                pipe.InitBuffer(out_queue_dgamma, 1, C_G_alloc_space);
                pipe.InitBuffer(out_queue_dbeta, 1, C_G_alloc_space);
                pipe.InitBuffer(temp_queue_HXW, 1, C_GxHxW_alloc_space);
                if (!(this->ele_num_per_channel % float_per_block == 0 || this->ele_num_per_group == 1)) {
                    pipe.InitBuffer(in_temp_x, HxW_alloc_space);
                    pipe.InitBuffer(in_temp_dy, HxW_alloc_space);
                }
            } else if (Tiling_key == MODE_1) {
                // VEC_IN
                uint32_t HxW_alloc_space = ceil(this->ele_num_per_channel, T_per_block) * sizeof(float);
                uint32_t NxG_alloc_space = T_per_block * sizeof(float);
                uint32_t C_G_alloc_space = ceil(this->C_G, T_per_block) * sizeof(float);
                pipe.InitBuffer(in_queue_dy, 1, HxW_alloc_space);
                pipe.InitBuffer(in_queue_x, 1, HxW_alloc_space);
                pipe.InitBuffer(in_queue_gamma, 1, C_G_alloc_space);
                // VEC_OUT
                pipe.InitBuffer(out_queue_dgamma, 1, C_G_alloc_space);
                pipe.InitBuffer(out_queue_dbeta, 1, C_G_alloc_space);
                pipe.InitBuffer(temp_queue_HXW, 1, HxW_alloc_space);
            } else if (Tiling_key == MODE_2) {
                // VEC_IN
                uint32_t HxW_alloc_space = ceil(this->ele_num_per_channel, T_per_block) * sizeof(float);
                uint32_t NxG_alloc_space = T_per_block * sizeof(float);
                uint32_t C_G_alloc_space = ceil(this->C_G, T_per_block) * sizeof(float);
                pipe.InitBuffer(in_queue_dy, doubleBuffer, HxW_alloc_space);
                pipe.InitBuffer(in_queue_x, doubleBuffer, HxW_alloc_space);
                pipe.InitBuffer(in_queue_gamma, 1, C_G_alloc_space);
                // VEC_OUT
                pipe.InitBuffer(out_queue_dgamma, doubleBuffer, C_G_alloc_space);
                pipe.InitBuffer(out_queue_dbeta, doubleBuffer, C_G_alloc_space);
                pipe.InitBuffer(temp_queue_HXW, doubleBuffer, HxW_alloc_space);
            } else if (Tiling_key == MODE_3) {
                this->mode2_ub_capacity_ele = tiling_data->mode2_ub_capacity_ele;
                this->mode2_ub_iteration_num = tiling_data->mode2_ub_iteration_num;
                this->mode2_ub_tail_num = tiling_data->mode2_ub_tail_num;
                // VEC_IN
                uint32_t capacity_ele_alloc_space = ceil(this->mode2_ub_capacity_ele, T_per_block) * sizeof(float);
                uint32_t NxG_alloc_space = T_per_block * sizeof(float);
                uint32_t C_G_alloc_space = ceil(this->C_G, T_per_block) * sizeof(float);
                pipe.InitBuffer(in_queue_dy, 1, capacity_ele_alloc_space);
                pipe.InitBuffer(in_queue_x, 1, capacity_ele_alloc_space);
                pipe.InitBuffer(in_queue_gamma, 1, C_G_alloc_space);
                pipe.InitBuffer(temp_queue_HXW, 1, capacity_ele_alloc_space);
                pipe.InitBuffer(out_queue_dgamma, 1, C_G_alloc_space);
                pipe.InitBuffer(out_queue_dbeta, 1, C_G_alloc_space);
            } else if (Tiling_key == MODE_5) {
                // VEC_IN
                uint32_t C_G_alloc_space = ceil(this->C_G, T_per_block) * sizeof(T);
                pipe.InitBuffer(in_queue_dy, 1, C_G_alloc_space);
                pipe.InitBuffer(in_queue_x, 1, C_G_alloc_space);
                pipe.InitBuffer(in_queue_gamma, 1, C_G_alloc_space);
                // VEC_OUT
                pipe.InitBuffer(out_queue_dgamma, 1, C_G_alloc_space);
                pipe.InitBuffer(out_queue_dbeta, 1, C_G_alloc_space);
                pipe.InitBuffer(temp_queue_HXW, 1, C_G_alloc_space);
            }
        } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
            if (Tiling_key == MODE_0) {
                uint32_t HxW_alloc_space = ceil(this->ele_num_per_channel, T_per_block);
                uint32_t C_GxHxW_alloc_space = ceil(this->ele_num_per_group, T_per_block);
                uint32_t NxG_alloc_space = T_per_block;
                uint32_t C_G_alloc_space = ceil(this->C_G, T_per_block);
                // VEC_IN
                pipe.InitBuffer(in_queue_dy, 1, C_GxHxW_alloc_space * sizeof(float));
                pipe.InitBuffer(in_tbuf_dy_T, C_GxHxW_alloc_space * sizeof(T));
                pipe.InitBuffer(in_queue_x, 1, C_GxHxW_alloc_space * sizeof(float));
                if (!(this->ele_num_per_channel % float_per_block == 0 || this->ele_num_per_group == 1)) {
                    pipe.InitBuffer(in_temp_x, HxW_alloc_space * sizeof(float));
                    pipe.InitBuffer(in_temp_dy, HxW_alloc_space * sizeof(float));
                }
                pipe.InitBuffer(in_tbuf_x_T, C_GxHxW_alloc_space * sizeof(T));
                pipe.InitBuffer(in_queue_gamma, 1, C_G_alloc_space * sizeof(float));
                pipe.InitBuffer(in_tbuf_gamma_T, C_G_alloc_space * sizeof(T));
                pipe.InitBuffer(out_tbuf_dgamma_T, C_G_alloc_space * sizeof(T));
                pipe.InitBuffer(out_tbuf_dbeta_T, C_G_alloc_space * sizeof(T));
                // VEC_OUT
                pipe.InitBuffer(out_queue_dgamma, 1, C_G_alloc_space * sizeof(float));
                pipe.InitBuffer(out_queue_dbeta, 1, C_G_alloc_space * sizeof(float));
                pipe.InitBuffer(temp_queue_HXW, 1, C_GxHxW_alloc_space * sizeof(float));
            } else if (Tiling_key == MODE_1) {
                // VEC_IN
                uint32_t HxW_alloc_space = ceil(this->ele_num_per_channel, T_per_block);
                uint32_t NxG_alloc_space = T_per_block;
                uint32_t C_G_alloc_space = ceil(this->C_G, T_per_block);
                pipe.InitBuffer(in_queue_dy, 1, HxW_alloc_space * sizeof(float));
                pipe.InitBuffer(in_tbuf_dy_T, HxW_alloc_space * sizeof(T));
                pipe.InitBuffer(in_queue_x, 1, HxW_alloc_space * sizeof(float));
                pipe.InitBuffer(in_tbuf_x_T, HxW_alloc_space * sizeof(T));
                pipe.InitBuffer(in_queue_gamma, 1, C_G_alloc_space * sizeof(float));
                pipe.InitBuffer(in_tbuf_gamma_T, C_G_alloc_space * sizeof(T));
                pipe.InitBuffer(out_tbuf_dgamma_T, C_G_alloc_space * sizeof(T));
                pipe.InitBuffer(out_tbuf_dbeta_T, C_G_alloc_space * sizeof(T));
                // VEC_OUT
                pipe.InitBuffer(out_queue_dgamma, 1, C_G_alloc_space * sizeof(float));
                pipe.InitBuffer(out_queue_dbeta, 1, C_G_alloc_space * sizeof(float));
                pipe.InitBuffer(temp_queue_HXW, 1, HxW_alloc_space * sizeof(float));
            } else if (Tiling_key == MODE_2) {
                // VEC_IN
                uint32_t HxW_alloc_space = ceil(this->ele_num_per_channel, T_per_block);
                uint32_t NxG_alloc_space = T_per_block;
                uint32_t C_G_alloc_space = ceil(this->C_G, T_per_block);
                pipe.InitBuffer(in_queue_dy, doubleBuffer, HxW_alloc_space * sizeof(float));
                pipe.InitBuffer(in_tbuf_dy_T, HxW_alloc_space * sizeof(T));
                pipe.InitBuffer(in_queue_x, doubleBuffer, HxW_alloc_space * sizeof(float));
                pipe.InitBuffer(in_tbuf_x_T, HxW_alloc_space * sizeof(T));
                pipe.InitBuffer(in_queue_gamma, 1, C_G_alloc_space * sizeof(float));
                pipe.InitBuffer(in_tbuf_gamma_T, C_G_alloc_space * sizeof(T));
                pipe.InitBuffer(out_tbuf_dgamma_T, C_G_alloc_space * sizeof(T));
                pipe.InitBuffer(out_tbuf_dbeta_T, C_G_alloc_space * sizeof(T));
                // VEC_OUT
                pipe.InitBuffer(out_queue_dgamma, doubleBuffer, C_G_alloc_space * sizeof(float));
                pipe.InitBuffer(out_queue_dbeta, doubleBuffer, C_G_alloc_space * sizeof(float));
                pipe.InitBuffer(temp_queue_HXW, doubleBuffer, HxW_alloc_space * sizeof(float));
            } else if (Tiling_key == MODE_3) {
                this->mode2_ub_capacity_ele = tiling_data->mode2_ub_capacity_ele;
                this->mode2_ub_iteration_num = tiling_data->mode2_ub_iteration_num;
                this->mode2_ub_tail_num = tiling_data->mode2_ub_tail_num;
                uint32_t capacity_ele_alloc_space = ceil(this->mode2_ub_capacity_ele, T_per_block);
                uint32_t NxG_alloc_space = T_per_block;
                uint32_t C_G_alloc_space = ceil(this->C_G, T_per_block);
                // VEC_IN
                pipe.InitBuffer(in_queue_dy, 1, capacity_ele_alloc_space * sizeof(float));
                pipe.InitBuffer(in_tbuf_dy_T, capacity_ele_alloc_space * sizeof(T));
                pipe.InitBuffer(in_queue_x, 1, capacity_ele_alloc_space * sizeof(float));
                pipe.InitBuffer(in_tbuf_x_T, capacity_ele_alloc_space * sizeof(T));
                pipe.InitBuffer(in_queue_gamma, 1, C_G_alloc_space * sizeof(float));
                pipe.InitBuffer(in_tbuf_gamma_T, C_G_alloc_space * sizeof(T));
                pipe.InitBuffer(out_tbuf_dgamma_T, C_G_alloc_space * sizeof(T));
                pipe.InitBuffer(out_tbuf_dbeta_T, C_G_alloc_space * sizeof(T));
                // VEC_OUT
                pipe.InitBuffer(out_queue_dgamma, 1, C_G_alloc_space * sizeof(float));
                pipe.InitBuffer(out_queue_dbeta, 1, C_G_alloc_space * sizeof(float));
                pipe.InitBuffer(temp_queue_HXW, 1, capacity_ele_alloc_space * sizeof(float));
            } else if (Tiling_key == MODE_5) {
                // VEC_IN
                uint32_t C_G_alloc_space = ceil(this->C_G, T_per_block);
                pipe.InitBuffer(in_queue_dy, 1, C_G_alloc_space * sizeof(float));
                pipe.InitBuffer(in_tbuf_dy_T, C_G_alloc_space * sizeof(T));
                pipe.InitBuffer(in_queue_x, 1, C_G_alloc_space * sizeof(float));
                pipe.InitBuffer(in_tbuf_x_T, C_G_alloc_space * sizeof(T));
                pipe.InitBuffer(in_queue_gamma, 1, C_G_alloc_space * sizeof(float));
                pipe.InitBuffer(in_tbuf_gamma_T, C_G_alloc_space * sizeof(T));
                pipe.InitBuffer(out_tbuf_dgamma_T, C_G_alloc_space * sizeof(T));
                pipe.InitBuffer(out_tbuf_dbeta_T, C_G_alloc_space * sizeof(T));
                // VEC_OUT
                pipe.InitBuffer(out_queue_dgamma, 1, C_G_alloc_space * sizeof(float));
                pipe.InitBuffer(out_queue_dbeta, 1, C_G_alloc_space * sizeof(float));
                pipe.InitBuffer(temp_queue_HXW, 1, C_G_alloc_space * sizeof(float));
            }
        }
    }

    __aicore__ inline void Get_Mean_Rstd(int32_t task_idx)
    {
        if constexpr (IsSameType<T, bfloat16_t>::value) {
            this->meanScalar = ToFloat(mean_gm.GetValue(task_idx));
            this->rstdScalar = ToFloat(rstd_gm.GetValue(task_idx));
        } else {
            this->meanScalar = float(mean_gm.GetValue(task_idx));
            this->rstdScalar = float(rstd_gm.GetValue(task_idx));
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Custom_DataCopy_In(const LocalTensor<float> &_in, TBuf<TPosition::VECCALC> &tbuf,
        const GlobalTensor<T> &gm, const uint32_t offset, const uint32_t count)
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(_in, gm[offset], copyParams, padParams);
        } else {
            LocalTensor<T> temp = tbuf.Get<T>(count);
            DataCopyPad(temp, gm[offset], copyParams, padParams);
            PipeBarrier<PIPE_ALL>();
            Cast(_in, temp, RoundMode::CAST_NONE, count);
        }
    }

    __aicore__ inline void Custom_DataCopy_In(LocalTensor<float> &_in, const uint32_t ub_offset,
        TBuf<TPosition::VECCALC> &tbuf, GlobalTensor<T> &gm, const uint32_t gm_offset, const uint32_t count)
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(_in[ub_offset], gm[gm_offset], copyParams, padParams);
        } else {
            LocalTensor<T> temp = tbuf.Get<T>(count);
            DataCopyPad(temp, gm[gm_offset], copyParams, padParams);
            PipeBarrier<PIPE_ALL>();
            Cast(_in[ub_offset], temp, RoundMode::CAST_NONE, count);
        }
    }

    __aicore__ inline void Custom_DataCopy_Out(
        LocalTensor<float> &_out, const uint32_t gm_offset, TBuf<TPosition::VECCALC> &tbuf, const uint32_t count)
    {
        DataCopyParams copyParams{1, (uint16_t)(count * sizeof(T)), 0, 0};
        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(dx_gm[gm_offset], _out, copyParams);
        } else {
            LocalTensor<T> temp = tbuf.Get<T>(count);
            Cast(temp, _out, RoundMode::CAST_ROUND, count);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            DataCopyPad(dx_gm[gm_offset], temp, copyParams);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

    __aicore__ inline void Custom_DataCopy_Out(LocalTensor<float> &_out, const uint32_t ub_offset,
        const uint32_t gm_offset, TBuf<TPosition::VECCALC> &tbuf, const uint32_t count)
    {
        DataCopyParams copyParams{1, (uint16_t)(count * sizeof(T)), 0, 0};
        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(dx_gm[gm_offset], _out[ub_offset], copyParams);
        } else {
            LocalTensor<T> temp = tbuf.Get<T>(count);
            Cast(temp, _out[ub_offset], RoundMode::CAST_ROUND, count);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyPad(dx_gm[gm_offset], temp, copyParams);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

    __aicore__ inline void Custom_DataCopy_Out(LocalTensor<float> &_out, GlobalTensor<T> &gm_out,
        const uint32_t gm_offset, TBuf<TPosition::VECCALC> &tbuf, const uint32_t count)
    {
        DataCopyParams copyParams{1, (uint16_t)(count * sizeof(T)), 0, 0};
        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(gm_out[gm_offset], _out, copyParams);
        } else {
            LocalTensor<T> temp = tbuf.Get<T>(count);
            Cast(temp, _out, RoundMode::CAST_ROUND, count);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyPad(gm_out[gm_offset], temp, copyParams);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

    __aicore__ inline void fp32_dgamma_dbeta2GM(uint32_t channel_idx, GlobalTensor<float> &dgamma_out,
        const LocalTensor<float> &dgamma_ub, GlobalTensor<float> &dbeta_out, const LocalTensor<float> &dbeta_ub)
    {
        event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
        SetAtomicAdd<float>();
        DataCopyParams copyParams{1, (uint16_t)(this->C_G * sizeof(float)), 0, 0};
        if (dbetaIsRequire) {
            DataCopyPad(dbeta_out[channel_idx], dbeta_ub, copyParams);
            event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
            WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
        }
        if (dgammaIsRequire) {
            DataCopyPad(dgamma_out[channel_idx], dgamma_ub, copyParams);
            event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
            WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
        }
        SetAtomicNone();
    }

    __aicore__ inline void non_fp32_dgamma_dbeta2GM(uint32_t channel_idx, GlobalTensor<T> &dgamma_out,
        const LocalTensor<float> &dgamma_ub, GlobalTensor<T> &dbeta_out, const LocalTensor<float> &dbeta_ub,
        TBuf<TPosition::VECCALC> &dgamma_tbuf, TBuf<TPosition::VECCALC> &dbeta_tbuf)
    {
        event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
        DataCopyParams copyParams{1, (uint16_t)(this->C_G * sizeof(T)), 0, 0};
        if (dbetaIsRequire) {
            LocalTensor<T> dbeta_temp = dbeta_tbuf.Get<T>(this->C_G);
            Cast(dbeta_temp, dbeta_ub, RoundMode::CAST_ROUND, this->C_G);
            event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            DataCopyPad(dbeta_out[channel_idx], dbeta_temp, copyParams);
        }
        if (dgammaIsRequire) {
            LocalTensor<T> dgamma_temp = dgamma_tbuf.Get<T>(this->C_G);
            Cast(dgamma_temp, dgamma_ub, RoundMode::CAST_ROUND, this->C_G);
            event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            DataCopyPad(dgamma_out[channel_idx], dgamma_temp, copyParams);
        }
    }

    __aicore__ inline void Mode_Selection(
        int32_t task_idx, const LocalTensor<float> &dgamma_ub, const LocalTensor<float> &dbeta_ub)
    {
        uint32_t outChannelIdx;
        constexpr uint32_t SPLIT_COUNT = 2;
        if constexpr (!isDeterministic && IsSameType<T, float>::value) {
            outChannelIdx = (task_idx % this->G) * this->C_G;
            fp32_dgamma_dbeta2GM(outChannelIdx, dgamma_gm, dgamma_ub, dbeta_gm, dbeta_ub);
        } else if constexpr (!isDeterministic && !IsSameType<T, float>::value) {
            outChannelIdx = (task_idx % this->G) * this->C_G;
            if (N == 1) {
                non_fp32_dgamma_dbeta2GM(
                    outChannelIdx, dgamma_gm, dgamma_ub, dbeta_gm, dbeta_ub, out_tbuf_dgamma_T, out_tbuf_dbeta_T);
            } else {
                fp32_dgamma_dbeta2GM(outChannelIdx, dgamma_workspace, dgamma_ub, dbeta_workspace, dbeta_ub);
            }
        } else if constexpr (isDeterministic) {
            outChannelIdx = ((task_idx / this->G) / SPLIT_COUNT * this->G + (task_idx % this->G)) * this->C_G;
            fp32_dgamma_dbeta2GM(outChannelIdx, dgamma_workspace, dgamma_ub, dbeta_workspace, dbeta_ub);
        }
    }

    __aicore__ inline uint32_t divceil(uint32_t a, uint32_t b)
    {
        if (b != 0) {
            return (a - 1) / b + 1;
        } else {
            return 0;
        }
    }

    __aicore__ inline uint32_t ceil(uint32_t a, uint32_t b)
    {
        if (b != 0) {
            return ((a - 1) / b + 1) * b;
        } else {
            return 0;
        }
    }

    __aicore__ inline uint32_t floor(uint32_t a, uint32_t b)
    {
        if (b != 0) {
            return (a / b * b);
        } else {
            return 0;
        }
    }

    __aicore__ inline void Adds_Muls_template(const LocalTensor<float> &dstLocal, const LocalTensor<float> &srcLocal,
        float adds_scalarValue, float muls_scalarValue, const int32_t calCount)
    {
        Muls(dstLocal, srcLocal, muls_scalarValue, calCount);
        Adds(dstLocal, srcLocal, adds_scalarValue, calCount);
    }

    __aicore__ inline void Adds_Muls_template(const LocalTensor<float> &mul_dstLocal,
        const LocalTensor<float> &mul_srcLocal, const LocalTensor<float> &add_dstLocal,
        const LocalTensor<float> &add_srcLocal, float adds_scalarValue, float muls_scalarValue, const int32_t calCount)
    {
        Muls(mul_dstLocal, mul_srcLocal, muls_scalarValue, calCount);
        Adds(add_dstLocal, add_srcLocal, adds_scalarValue, calCount);
    }

    __aicore__ inline float ReduceSum_template(const LocalTensor<float> &dstLocal, const LocalTensor<float> &srcLocal,
        const LocalTensor<float> &workLocal, uint32_t calCount, bool mode)
    {
        float temp_value = 0;
        const int32_t MASK = 64;
        const int32_t MAX_LEN = 128;
        const uint32_t STRIDE_BLOCK = 1;
        const uint8_t PROCESS_BLOCK = 8;
        if (mode && calCount > MASK) {
            bool firstAdd = true;
            if (calCount > MASK) {
                for (int32_t start = MAX_LEN; start >= 1; start >>= 1) {
                    const int32_t innerOffset = start * MASK;
                    if (calCount > innerOffset && firstAdd) {
                        Copy(dstLocal,
                            srcLocal,
                            MASK,
                            start,
                            {STRIDE_BLOCK, STRIDE_BLOCK, PROCESS_BLOCK, PROCESS_BLOCK});
                        Add<float>(dstLocal, srcLocal[innerOffset], dstLocal, calCount - innerOffset);
                        calCount = innerOffset;
                        firstAdd = false;
                    } else if (calCount > innerOffset) {
                        Add<float, false>(dstLocal,
                            dstLocal[innerOffset],
                            dstLocal,
                            MASK,
                            start,
                            {STRIDE_BLOCK, STRIDE_BLOCK, STRIDE_BLOCK, PROCESS_BLOCK, PROCESS_BLOCK, PROCESS_BLOCK});
                        calCount = innerOffset;
                    }
                }
                ReduceSum<float>(dstLocal, dstLocal, workLocal, MASK);
                temp_value = dstLocal.GetValue(0);
            }
        } else {
            ReduceSum<float>(dstLocal, srcLocal, workLocal, calCount);
            temp_value = dstLocal.GetValue(0);
        }
        return temp_value;
    }

    __aicore__ inline void Mul_ReduceSum_template(const LocalTensor<float> &dstLocal1,
        const LocalTensor<float> &dstLocal2, const LocalTensor<float> &srcLocal, const int32_t calCount, float &res1,
        float &res2)
    {
        Mul(dstLocal1, srcLocal, dstLocal1, calCount);
        Mul(dstLocal2, srcLocal, dstLocal2, calCount);
        res1 = ReduceSum_template(dstLocal1, dstLocal1, dstLocal1, calCount, false) / this->ele_num_per_group;
        res2 = ReduceSum_template(dstLocal2, dstLocal2, dstLocal2, calCount, false) / this->ele_num_per_group;
    }

    __aicore__ inline void Initbuffer4stage2_mode0()
    {
        pipe.InitBuffer(in_queue_dgamma_channel, 1, castEleNum * sizeof(float));
        pipe.InitBuffer(in_queue_dbeta_channel, 1, castEleNum * sizeof(float));
        pipe.InitBuffer(out_tbuf_dgamma_channel_T, castEleNum * sizeof(T));
        pipe.InitBuffer(out_tbuf_dbeta_channel_T, castEleNum * sizeof(T));
    }

    __aicore__ inline void Initbuffer4stage2_mode1()
    {
        pipe.InitBuffer(cal_queue_dgamma_reduce, 1, castEleNum * sizeof(float));
        pipe.InitBuffer(cal_queue_dbeta_reduce, 1, castEleNum * sizeof(float));
        pipe.InitBuffer(in_queue_dgamma_channel, 1, coreBatchParts * castEleNum * sizeof(float));
        pipe.InitBuffer(in_queue_dbeta_channel, 1, coreBatchParts * castEleNum * sizeof(float));
    }

    __aicore__ inline void Initbuffer4stage2_mode2()
    {
        pipe.InitBuffer(cal_queue_dgamma_reduce, 1, castEleNum * sizeof(float));
        pipe.InitBuffer(cal_queue_dbeta_reduce, 1, castEleNum * sizeof(float));
        pipe.InitBuffer(in_queue_dgamma_channel, 1, coreBatchParts * castEleNum * sizeof(float));
        pipe.InitBuffer(in_queue_dbeta_channel, 1, coreBatchParts * castEleNum * sizeof(float));
        pipe.InitBuffer(out_tbuf_dgamma_channel_T, castEleNum * sizeof(T));
        pipe.InitBuffer(out_tbuf_dbeta_channel_T, castEleNum * sizeof(T));
    }

    __aicore__ inline void cast_dgamma_dbeta_WSP2GM(uint32_t channel_idx, uint32_t count)
    {
        DataCopyExtParams copyParams_fp16{1, (uint16_t)(count * sizeof(half)), 0, 0, 0};
        DataCopyExtParams copyParams_fp32{1, (uint16_t)(count * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        if (dgammaIsRequire) {
            LocalTensor<float> dgamma_Local = in_queue_dgamma_channel.AllocTensor<float>();
            DataCopyPad(dgamma_Local, dgamma_workspace[channel_idx], copyParams_fp32, padParams);
            LocalTensor<T> dgamma_temp = out_tbuf_dgamma_channel_T.Get<T>(count);
            event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            Cast(dgamma_temp, dgamma_Local, RoundMode::CAST_ROUND, count);
            in_queue_dgamma_channel.FreeTensor(dgamma_Local);
            event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            DataCopyPad(dgamma_gm[channel_idx], dgamma_temp, copyParams_fp16);
        }
        if (dbetaIsRequire) {
            LocalTensor<float> dbeta_Local = in_queue_dbeta_channel.AllocTensor<float>();
            DataCopyPad(dbeta_Local, dbeta_workspace[channel_idx], copyParams_fp32, padParams);
            LocalTensor<T> dbeta_temp = out_tbuf_dbeta_channel_T.Get<T>(count);
            event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            Cast(dbeta_temp, dbeta_Local, RoundMode::CAST_ROUND, count);
            in_queue_dbeta_channel.FreeTensor(dbeta_Local);
            event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            DataCopyPad(dbeta_gm[channel_idx], dbeta_temp, copyParams_fp16);
        }
    }

    __aicore__ inline void reduce_axis_n_WSP2UB(TQue<QuePosition::VECIN, 1> &vecInQue,
        const GlobalTensor<float> &workspace, uint32_t workSpaceOffset, uint32_t repeatTime, uint32_t count,
        const LocalTensor<float> &dstLocal)
    {
        LocalTensor<float> vecInLocal = vecInQue.AllocTensor<float>();
        DataCopyExtParams copyParams_in{(uint16_t)repeatTime,
            (uint16_t)(count * sizeof(float)),
            (uint16_t)((C - count) * sizeof(float)),
            (uint16_t)((castEleNum - count) / float_per_block),
            0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(vecInLocal, workspace[workSpaceOffset], copyParams_in, padParams);
        event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        int32_t end = repeatTime;
        int32_t mid = repeatTime;
        constexpr int MIN_LOOP_THRESHOLD = 2;
        while (mid >= MIN_LOOP_THRESHOLD) {
            mid >>= 1;
            end -= mid;
            PipeBarrier<PIPE_V>();
            Add(vecInLocal, vecInLocal[end * castEleNum], vecInLocal, mid * castEleNum);
            PipeBarrier<PIPE_V>();
            mid = end;
        }
        PipeBarrier<PIPE_V>();
        Add(dstLocal, vecInLocal, dstLocal, count);
        vecInQue.FreeTensor(vecInLocal);
    }

    __aicore__ inline void reduce_dgamma_dbeta_WSP2GM(
        uint32_t start_offset, uint32_t channel_idx, uint32_t count, uint32_t reduce_axis_num)
    {
        uint32_t repeatTime = 0;
        LocalTensor<float> dbeta_sum_Local = cal_queue_dbeta_reduce.AllocTensor<float>();
        LocalTensor<float> dgamma_sum_Local = cal_queue_dgamma_reduce.AllocTensor<float>();
        Duplicate<float>(dbeta_sum_Local, 0.0f, castEleNum);
        Duplicate<float>(dgamma_sum_Local, 0.0f, castEleNum);
        PipeBarrier<PIPE_V>();
        for (uint32_t loop_idx = 0; loop_idx < divceil(reduce_axis_num, coreBatchParts); loop_idx++) {
            uint32_t loop_offset = loop_idx * coreBatchParts * C;
            if (loop_idx < divceil(reduce_axis_num, coreBatchParts) - 1) {
                repeatTime = coreBatchParts;
            } else if (loop_idx == divceil(reduce_axis_num, coreBatchParts) - 1) {
                repeatTime = reduce_axis_num % coreBatchParts == 0 ? coreBatchParts : reduce_axis_num % coreBatchParts;
            }
            if (dgammaIsRequire) {
                reduce_axis_n_WSP2UB(in_queue_dgamma_channel,
                    dgamma_workspace,
                    start_offset + loop_offset + channel_idx,
                    repeatTime,
                    count,
                    dgamma_sum_Local);
            }
            if (dbetaIsRequire) {
                reduce_axis_n_WSP2UB(in_queue_dbeta_channel,
                    dbeta_workspace,
                    start_offset + loop_offset + channel_idx,
                    repeatTime,
                    count,
                    dbeta_sum_Local);
            }
        }
        event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        DataCopyExtParams copyParams_out{1, (uint16_t)(count * sizeof(float)), 0, 0, 0};
        if (dgammaIsRequire) {
            SetAtomicAdd<float>();
            DataCopyPad(dgamma_gm[channel_idx], dgamma_sum_Local, copyParams_out);
            SetAtomicNone();
        }
        cal_queue_dgamma_reduce.FreeTensor(dgamma_sum_Local);
        if (dbetaIsRequire) {
            SetAtomicAdd<float>();
            DataCopyPad(dbeta_gm[channel_idx], dbeta_sum_Local, copyParams_out);
            SetAtomicNone();
        }
        cal_queue_dbeta_reduce.FreeTensor(dbeta_sum_Local);
    }

    __aicore__ inline void reduce_cast_dgamma_dbeta_WSP2GM(
        uint32_t channel_idx, uint32_t count, uint32_t reduce_axis_num)
    {
        uint32_t repeatTime = 0;
        LocalTensor<float> dbeta_sum_Local = cal_queue_dbeta_reduce.AllocTensor<float>();
        LocalTensor<float> dgamma_sum_Local = cal_queue_dgamma_reduce.AllocTensor<float>();
        Duplicate<float>(dbeta_sum_Local, 0.0f, castEleNum);
        Duplicate<float>(dgamma_sum_Local, 0.0f, castEleNum);
        PipeBarrier<PIPE_V>();
        for (uint32_t loop_idx = 0; loop_idx < divceil(reduce_axis_num, coreBatchParts); loop_idx++) {
            uint32_t loop_offset = loop_idx * coreBatchParts * C;
            if (loop_idx < divceil(reduce_axis_num, coreBatchParts) - 1) {
                repeatTime = coreBatchParts;
            } else if (loop_idx == divceil(reduce_axis_num, coreBatchParts) - 1) {
                repeatTime = reduce_axis_num % coreBatchParts == 0 ? coreBatchParts : reduce_axis_num % coreBatchParts;
            }
            if (dgammaIsRequire) {
                reduce_axis_n_WSP2UB(in_queue_dgamma_channel,
                    dgamma_workspace,
                    loop_offset + channel_idx,
                    repeatTime,
                    count,
                    dgamma_sum_Local);
            }
            if (dbetaIsRequire) {
                reduce_axis_n_WSP2UB(in_queue_dbeta_channel,
                    dbeta_workspace,
                    loop_offset + channel_idx,
                    repeatTime,
                    count,
                    dbeta_sum_Local);
            }
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams copyParams_out{1, (uint16_t)(count * sizeof(T)), 0, 0, 0};
        if (dgammaIsRequire) {
            Custom_DataCopy_Out(dgamma_sum_Local, dgamma_gm, channel_idx, out_tbuf_dgamma_channel_T, count);
        }
        cal_queue_dgamma_reduce.FreeTensor(dgamma_sum_Local);
        if (dbetaIsRequire) {
            Custom_DataCopy_Out(dbeta_sum_Local, dbeta_gm, channel_idx, out_tbuf_dbeta_channel_T, count);
        }
        cal_queue_dbeta_reduce.FreeTensor(dbeta_sum_Local);
    }
};
#endif  // GROUP_NORM_GRAD