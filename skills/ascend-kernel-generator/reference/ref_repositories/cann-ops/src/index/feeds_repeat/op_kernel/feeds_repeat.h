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
 * \file feeds_repeat.h
 * \brief
 */
#ifndef FEEDS_REPEAT_H
#define FEEDS_REPEAT_H
#include "kernel_operator.h"

namespace FeedsRepeat{
using namespace AscendC;

template <typename T1, typename T2> 
class FeedsRepeatND{
public:
    __aicore__ inline FeedsRepeatND() {}
    __aicore__ inline void Init(GM_ADDR feeds, GM_ADDR feeds_repeat_times, GM_ADDR y, const FeedsRepeatTilingData* tiling_data);
    __aicore__ inline void Process();

private:
    __aicore__ inline void RepeatTimesCast();
    __aicore__ inline void ClearOutputSpace();
    __aicore__ inline void ComputeStartDest();
    __aicore__ inline void RepeatSingleRow();
    __aicore__ inline void RepeatMultiRow();

protected:
    TPipe pipe;
    GlobalTensor<T1> feeds_gm;
    GlobalTensor<T2> feeds_repeat_times_gm;
    GlobalTensor<T1> y_gm;

    TQue<QuePosition::VECIN,1> in_queue;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> in_out_queue;

    TBuf<TPosition::VECCALC> feeds_repeat_times_float_buf;
    TBuf<TPosition::VECCALC> end_sum_buf;
    TBuf<TPosition::VECCALC> end_sum_int64_buf;
    TBuf<TPosition::VECCALC> sum_result_buf;
    TBuf<TPosition::VECCALC> sum_result_int64_buf;

    LocalTensor<T1> row;
    LocalTensor<T2> feeds_repeat_times_ub;
    LocalTensor<float> feeds_repeat_times_float;
    LocalTensor<float> end_sum;
    LocalTensor<float> sum_result;
    LocalTensor<int64_t> end_sum_int64;
    LocalTensor<int64_t> sum_result_int64;

    uint32_t length;
    uint32_t length_aligned;
    uint32_t start_row;
    uint32_t end_row;
    int64_t elem_row;
    int64_t elem_per_loop;
    int64_t core_index;
    int64_t max_core_num;
    int64_t core_per_group = 0;
    int64_t core_moreover = 0;
    int64_t empty_size;
    int64_t row_per_core;
    int64_t row_left;
    int64_t repeat_times = 0;
    int64_t y_start_index = 0;
    int64_t end_index = 0;
    int64_t group_num = 0;
    int64_t group_id = -1;
    int64_t block_in_group = 1;
    int64_t id_in_group = -1;
    int64_t index;
    uint32_t start_aligned;
    int64_t loop_num;
    int64_t loop_left;
    int64_t start_index;
    int64_t elem_loop;
    int64_t left_index;
    const int64_t align_num = 32;
    
    event_t event_v_to_s;
    event_t event_mte3_to_v;
    DataCopyExtParams copyParams{1, 0, 0, 0, 0};
    DataCopyExtParams copyParams_repeat{1, 0, 0, 0, 0};
    DataCopyExtParams copyParams_left{1, 0, 0, 0, 0};
    DataCopyPadExtParams<T1> padParams{false, 0, 0, 0};
    DataCopyPadExtParams<T2> padParams_repeat{false, 0, 0, 0};
    SumParams sumParams{1, 0, 0};
    SumParams sumParams_total{1, 0, 0};
};

template <typename T1, typename T2> 
__aicore__ inline void FeedsRepeatND<T1, T2>::Init(GM_ADDR feeds, GM_ADDR feeds_repeat_times, GM_ADDR y, const FeedsRepeatTilingData* tiling_data){
    elem_row = tiling_data->elem_row;
    elem_per_loop = tiling_data->elem_per_loop;
    length = tiling_data->length;
    length_aligned = tiling_data->length_aligned;
    max_core_num = tiling_data->max_core_num;
    core_per_group = tiling_data->core_per_group;
    core_moreover = tiling_data->core_moreover;
    empty_size = tiling_data->empty_size;
    row_per_core = tiling_data->row_per_core;
    row_left = tiling_data->row_left;
    core_index = GetBlockIdx();

    feeds_gm.SetGlobalBuffer((__gm__ T1*)feeds);
    feeds_repeat_times_gm.SetGlobalBuffer((__gm__ T2*)feeds_repeat_times);
    y_gm.SetGlobalBuffer((__gm__ T1*)y);

    pipe.InitBuffer(in_out_queue, 2, elem_per_loop * sizeof(T1));
    pipe.InitBuffer(in_queue, 1, length_aligned * sizeof(T2));
    pipe.InitBuffer(feeds_repeat_times_float_buf, length_aligned * sizeof(float));
    pipe.InitBuffer(end_sum_buf, align_num);
    pipe.InitBuffer(end_sum_int64_buf, align_num);
    pipe.InitBuffer(sum_result_buf, align_num);
    pipe.InitBuffer(sum_result_int64_buf, align_num);

    event_mte3_to_v = static_cast<event_t>(pipe.FetchEventID<HardEvent::MTE3_V>());
    event_v_to_s = static_cast<event_t>(pipe.FetchEventID<HardEvent::V_S>());
}

template <typename T1, typename T2> 
__aicore__ inline void FeedsRepeatND<T1, T2>::RepeatTimesCast(){
    copyParams_repeat.blockLen = length * (uint32_t)sizeof(T2);
    feeds_repeat_times_ub = in_queue.AllocTensor<T2>();
    DataCopyPad(feeds_repeat_times_ub, feeds_repeat_times_gm, copyParams_repeat, padParams_repeat);
    in_queue.EnQue(feeds_repeat_times_ub);
    feeds_repeat_times_ub = in_queue.DeQue<T2>();
    Cast(feeds_repeat_times_float, feeds_repeat_times_ub, RoundMode::CAST_RINT, length_aligned);
    in_queue.FreeTensor(feeds_repeat_times_ub);
}

template <typename T1, typename T2> 
__aicore__ inline void FeedsRepeatND<T1, T2>::ClearOutputSpace(){
    end_sum = end_sum_buf.Get<float>();
    end_sum_int64 = end_sum_int64_buf.Get<int64_t>();
    //计算清零起始地址
    sumParams_total.inner = length_aligned;
    sumParams_total.n = length;
    Sum(end_sum, feeds_repeat_times_float, sumParams_total);
    Cast(end_sum_int64, end_sum, RoundMode::CAST_RINT, align_num);
    SetFlag<HardEvent::V_S>(event_v_to_s);
    WaitFlag<HardEvent::V_S>(event_v_to_s);
    end_index += end_sum_int64.GetValue(0);
    //gm清零
    int64_t empty_per_core = ((empty_size - end_index) * elem_row) / max_core_num;
    int64_t empty_left = ((empty_size - end_index) * elem_row) % max_core_num;
    if((core_index == 0) && ((empty_per_core + empty_left) != 0)){
        InitOutput<T1>(y_gm[end_index * elem_row], empty_per_core + empty_left, 0);
    }
    if ((core_index != 0) && (empty_per_core != 0)){
        InitOutput<T1>(y_gm[end_index * elem_row + core_index * empty_per_core + empty_left], empty_per_core, 0);
    }
    SetFlag<HardEvent::MTE3_V>(event_mte3_to_v);
    WaitFlag<HardEvent::MTE3_V>(event_mte3_to_v);
}

template <typename T1, typename T2> 
__aicore__ inline void FeedsRepeatND<T1, T2>::ComputeStartDest(){
    sum_result = sum_result_buf.Get<float>();
    sum_result_int64 = sum_result_int64_buf.Get<int64_t>();
    //计算搬运起始地址
    if(start_row != 0){
        sumParams.inner = start_aligned;
        sumParams.n = start_row;
        Sum(sum_result, feeds_repeat_times_float, sumParams);
        Cast(sum_result_int64, sum_result, RoundMode::CAST_RINT, align_num);
        SetFlag<HardEvent::V_S>(event_v_to_s);
        WaitFlag<HardEvent::V_S>(event_v_to_s);
        y_start_index += sum_result_int64.GetValue(0);
    }
}

template <typename T1, typename T2> 
__aicore__ inline void FeedsRepeatND<T1, T2>::RepeatSingleRow(){
    feeds_repeat_times_float = feeds_repeat_times_float_buf.Get<float>();
    repeat_times = feeds_repeat_times_ub.GetValue(start_row);
    int64_t repeat_left = repeat_times % block_in_group;
    int64_t repeat_start = id_in_group * (repeat_times / block_in_group) + (id_in_group < repeat_left ? id_in_group : repeat_left);
    int64_t repeat_end = repeat_start + (repeat_times / block_in_group) + (id_in_group < repeat_left ? 1 : 0);
    for(int j = 0; j < loop_num; j++){
        row = in_out_queue.AllocTensor<T1>();
        DataCopyPad(row, feeds_gm[start_index + elem_per_loop * j], copyParams, padParams);
        in_out_queue.EnQue<QuePosition::VECIN, QuePosition::VECOUT, T1>(row);
        row = in_out_queue.DeQue<QuePosition::VECIN, QuePosition::VECOUT, T1>();
        for(int k = repeat_start; k < repeat_end; k++){
            DataCopyPad(y_gm[(y_start_index + k) * elem_row + elem_per_loop * j], row, copyParams);
        }
        in_out_queue.FreeTensor(row);
    }
    if(loop_left != 0){
        copyParams_left.blockLen = (uint32_t)(loop_left * sizeof(T1));
        row = in_out_queue.AllocTensor<T1>();
        DataCopyPad(row, feeds_gm[left_index], copyParams_left, padParams);
        in_out_queue.EnQue<QuePosition::VECIN, QuePosition::VECOUT, T1>(row);
        row = in_out_queue.DeQue<QuePosition::VECIN, QuePosition::VECOUT, T1>();
        for(int k = repeat_start; k < repeat_end; k++){
            DataCopyPad(y_gm[(y_start_index + k) * elem_row + elem_loop], row, copyParams_left);
        }
        in_out_queue.FreeTensor(row);
    }
}

template <typename T1, typename T2> 
__aicore__ inline void FeedsRepeatND<T1, T2>::RepeatMultiRow(){
    int64_t loop_start;
    for(int i = start_row; i < end_row; i++){
        loop_start = elem_row * i;
        repeat_times = feeds_repeat_times_ub.GetValue(i);
        for(int j = 0; j < loop_num; j++){
            row = in_out_queue.AllocTensor<T1>();
            DataCopyPad(row, feeds_gm[loop_start + elem_per_loop * j], copyParams, padParams);
            in_out_queue.EnQue<QuePosition::VECIN, QuePosition::VECOUT, T1>(row);
            row = in_out_queue.DeQue<QuePosition::VECIN, QuePosition::VECOUT, T1>();
            for(int k = 0; k < repeat_times; k++){
                DataCopyPad(y_gm[(y_start_index + k) * elem_row + elem_per_loop * j], row, copyParams);
            }
            in_out_queue.FreeTensor(row);
        }
        if(loop_left != 0){
            copyParams_left.blockLen = (uint32_t)(loop_left * sizeof(T1));
            row = in_out_queue.AllocTensor<T1>();
            DataCopyPad(row, feeds_gm[loop_start + elem_loop], copyParams_left, padParams);
            in_out_queue.EnQue<QuePosition::VECIN, QuePosition::VECOUT, T1>(row);
            row = in_out_queue.DeQue<QuePosition::VECIN, QuePosition::VECOUT, T1>();
            for(int k = 0; k < repeat_times; k++){
                DataCopyPad(y_gm[(y_start_index + k) * elem_row + elem_loop], row, copyParams_left);
            }
            in_out_queue.FreeTensor(row);
        }
        y_start_index += repeat_times;
    }
}

template <typename T1, typename T2> 
__aicore__ inline void FeedsRepeatND<T1, T2>::Process(){
    if (core_per_group != 0){ 
        //行数大于核数
        group_num = (max_core_num - core_moreover) / core_per_group;
        if(core_index < core_moreover * (core_per_group +1)){
            block_in_group = core_per_group + 1;
            group_id = core_index / block_in_group;
            id_in_group = core_index % block_in_group;
        }
        else{
            block_in_group = core_per_group;
            group_id = (core_index - core_moreover) / core_per_group;
            id_in_group = (core_index - core_moreover) % core_per_group;
        }
    }
    //feeds分核
    index = core_per_group == 0 ? core_index : group_id;
    start_row = index * row_per_core + (index < row_left ? index : row_left);
    end_row = start_row + row_per_core + (index < row_left ? 1 : 0);
    start_aligned = (start_row * sizeof(float) + align_num - 1) / align_num * align_num / sizeof(float);
    //feeds一行拆分
    loop_num = elem_row / elem_per_loop;
    loop_left = elem_row % elem_per_loop;
    start_index = elem_row * start_row;
    elem_loop = elem_per_loop * loop_num;
    left_index = start_index + elem_loop;
    feeds_repeat_times_float = feeds_repeat_times_float_buf.Get<float>();
    RepeatTimesCast();
    ClearOutputSpace();
    ComputeStartDest();
    if(elem_row <= elem_per_loop){
        copyParams.blockLen = (uint32_t)(elem_row * sizeof(T1));
    }
    else{
        copyParams.blockLen = (uint32_t)(elem_per_loop * sizeof(T1));
    }
    if (core_per_group != 0){   //一行多核
        RepeatSingleRow();
    }
    else{   //一核多行
        RepeatMultiRow();
    }
}

} //namespace FeedsRepeat
#endif //FEEDS_REPEAT_H