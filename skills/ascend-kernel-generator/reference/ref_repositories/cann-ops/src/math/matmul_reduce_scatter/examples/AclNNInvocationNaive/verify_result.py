#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
import sys
import numpy as np
import torch


DEBUG_SWITCH = False


class Result:
    def __init__(self, result_name, total_big_num=0, total_big_ratio=0, diff_big_max=0, diff_big_avg=0, diff_big_sum=0,
                 total_small_num=0, total_small_ratio=0, err_small_num=0, err_small_ratio=0,
                 diff_rmse=0, rst_eb=0, diff_eb=0,
                 num_total_nan=0, err_total_nan=0, num_total_inf=0, err_total_inf=0, num_total_ninf=0,
                 err_total_ninf=0):
        self.result_name = result_name
        self.total_big_num = total_big_num
        self.total_big_ratio = total_big_ratio
        self.diff_big_max = diff_big_max
        self.diff_big_avg = diff_big_avg
        self.diff_big_sum = diff_big_sum
        self.total_small_num = total_small_num
        self.total_small_ratio = total_small_ratio
        self.err_small_num = err_small_num
        self.err_small_ratio = err_small_ratio
        self.diff_rmse = diff_rmse
        self.rst_eb = rst_eb
        self.diff_eb = diff_eb
        self.num_total_nan = num_total_nan
        self.err_total_nan = err_total_nan
        self.num_total_inf = num_total_inf
        self.err_total_inf = err_total_inf
        self.num_total_ninf = num_total_ninf
        self.err_total_ninf = err_total_ninf

    # 打印精度结果细节
    def print_result(self):
        print(f"正在打印结果：{self.result_name}")
        print(f" 大值总数：{self.total_big_num}")
        print(f" 大值占比：{self.total_big_ratio:.2%}")
        print(f" 大值最大误差：{self.diff_big_max:.8f}")
        print(f" 大值平均误差：{self.diff_big_avg:.8f}")
        print(f" 大值误差总和：{self.diff_big_sum:.2f}")
        print(f" 小值总数：{self.total_small_num}")
        print(f" 小值占比：{self.total_small_ratio:.2%}")
        print(f" 小值错误数：{self.err_small_num}，占比{self.err_small_ratio:.2%}")
        print(f" 误差均方根（RMSE）：{self.diff_rmse:.8f}")
        print(f" 均衡性偏差计数：{self.rst_eb}")
        print(f" 均衡性diff总和：{self.diff_eb:.8f}")
        if (self.num_total_nan + self.num_total_inf + self.num_total_ninf != 0) or \
                (self.err_total_nan + self.err_total_inf + self.err_total_ninf != 0) or True:
            print(f" golden nan总数：{self.num_total_nan}")
            print(f" nan误差数：{self.err_total_nan}")
            print(f" golden inf总数：{self.num_total_inf}")
            print(f" inf误差数：{self.err_total_inf}")
            print(f" golden -inf总数：{self.num_total_ninf}")
            print(f" -inf误差数：{self.err_total_ninf}")

    # 解析精度报错细节
    def check_result_debug(self, benchmark):
        reason_str = ''
        # Check diff_big_max conditions
        if self.diff_big_max > benchmark.diff_big_max * 10:
            reason_str += ' diff_big_max error,'
        elif self.diff_big_max > benchmark.diff_big_max:
            reason_str += ' diff_big_max warning,'
            
        # Check diff_big_avg conditions
        if self.diff_big_avg > benchmark.diff_big_avg * 2:
            reason_str += ' diff_big_avg error,'
        elif self.diff_big_avg > benchmark.diff_big_avg:
            reason_str += ' diff_big_avg warning,'

        if self.diff_big_sum > benchmark.diff_big_sum * 2:
            reason_str += ' diff_big_sum error,'
        elif self.diff_big_sum > benchmark.diff_big_sum:
            reason_str += ' diff_big_sum warning,'

        if self.err_small_num > benchmark.err_small_num * 2:
            reason_str += ' err_small_num error,'
        elif self.err_small_num > benchmark.err_small_num:
            reason_str += ' err_small_num warning,'

        if self.diff_rmse > benchmark.diff_rmse * 2:
            reason_str += ' diff_rmse error,'
        elif self.diff_rmse > benchmark.diff_rmse:
            reason_str += ' diff_rmse warning,'

        if self.err_total_nan > benchmark.err_total_nan:
            reason_str += ' err_total_nan error,'
        elif self.err_total_nan > 0:
            reason_str += ' err_total_nan warning,'

        if self.err_total_inf > benchmark.err_total_inf or self.err_total_ninf > benchmark.err_total_ninf:
            reason_str += ' err_total_inf error,'
        elif self.err_total_inf > 0 or self.err_total_ninf > 0:
            reason_str += ' err_total_inf warning,'

        return reason_str

    # 与竞品对比精度结果，benchmark传入gpu竞品数据或基线版本数据，返回检查结果与检查不通过原因
    def check_result(self, benchmark):
        conditions = [
            self.diff_big_max > benchmark.diff_big_max * 10,
            self.diff_big_avg > benchmark.diff_big_avg * 2,
            self.diff_big_sum > benchmark.diff_big_sum * 2,
            self.err_small_num > benchmark.err_small_num * 2,
            self.diff_rmse > benchmark.diff_rmse * 2
        ]
        if any(conditions):
            reason_str = self.check_result_debug(benchmark)
            return 'error', reason_str

        return 'ok', ''


def validate_tensor_shapes(value, golden, name):
    if value.shape != golden.shape:
        print(f"error: {name}计算结果错误，shape与标杆不匹配，用例执行失败！！！")
        print(f"debug: 输入shape {value.shape}")
        print(f"debug: 真值shape  {golden.shape}")
        return False
    return True


def handle_special_values(value, golden):
    # 处理INF/NAN特殊值
    value = value.clone()
    golden = golden.clone()
    
    # 记录特殊值统计
    nan_stats = {
        'num_total_nan': torch.sum(torch.isnan(golden)),
        'err_total_nan': torch.sum(torch.isnan(golden).logical_xor(torch.isnan(value)))
    }
    
    inf_mask = golden == torch.inf
    ninf_mask = golden == -torch.inf
    golden = torch.where(inf_mask, torch.finfo(value.dtype).max, golden)
    golden = torch.where(ninf_mask, torch.finfo(value.dtype).min, golden)
    value = torch.where(value == torch.inf, torch.finfo(value.dtype).max, value)
    value = torch.where(value == -torch.inf, torch.finfo(value.dtype).min, value)
    
    return value, golden, nan_stats


def process_large_values(value, golden, small_value):
    mask = golden >= small_value
    total_big_num = torch.sum(mask)
    
    value_big = value.clone()
    value_big[~mask] = 1
    golden_big = golden.clone()
    golden_big[~mask] = 1
    
    diff_big = torch.abs(value_big - golden_big)
    return {
        'total_big_num': total_big_num,
        'diff_big_max': diff_big.max(),
        'diff_big_sum': diff_big.sum(),
        'diff_big_avg': diff_big.sum() / total_big_num if total_big_num > 0 else 0
    }


def process_small_values(value, golden, small_value, small_value_atol):
    mask = golden < small_value
    total_small_num = torch.sum(mask)
    
    value_small = value.clone()
    value_small[~mask] = 1
    golden_small = golden.clone()
    golden_small[~mask] = 1
    
    diff_small = torch.abs(value_small - golden_small)
    err_small_num = torch.sum(diff_small > small_value_atol)
    return {
        'total_small_num': total_small_num,
        'err_small_num': err_small_num
    }


def calculate_rmse(value, golden):
    diff = torch.abs(value - golden)
    return torch.sqrt(torch.mean(torch.square(diff)))


def calculate_error_balance(value, golden):
    eb_bigger = torch.sum(value > golden)
    eb_smaller = torch.sum(value < golden)
    return {
        'rst_eb': torch.abs(eb_bigger - eb_smaller),
        'diff_eb': torch.sum(value - golden)
    }


def verify_tensor_result(value, golden, name):
    if not validate_tensor_shapes(value, golden, name):
        return None

    value, golden, nan_stats = handle_special_values(value, golden)
    
    dtype = value.dtype
    small_value = 0.001 if dtype in [torch.float16, torch.bfloat16] else 0.000001
    small_value_atol = 0.00001 if dtype in [torch.float16, torch.bfloat16] else 0.000000001

    big_data = process_large_values(value, golden, small_value)
    small_data = process_small_values(value, golden, small_value, small_value_atol)
    rmse = calculate_rmse(value, golden)
    eb_data = calculate_error_balance(value, golden)

    return Result(name,
        total_big_num=big_data['total_big_num'],
        total_big_ratio=big_data['total_big_num'] / golden.numel(),
        diff_big_max=big_data['diff_big_max'],
        diff_big_avg=big_data['diff_big_avg'],
        diff_big_sum=big_data['diff_big_sum'],
        total_small_num=small_data['total_small_num'],
        total_small_ratio=small_data['total_small_num'] / golden.numel(),
        err_small_num=small_data['err_small_num'],
        err_small_ratio=small_data['err_small_num'] / small_data['total_small_num'] 
            if small_data['total_small_num'] > 0 else 0,
        diff_rmse=rmse,
        rst_eb=eb_data['rst_eb'],
        diff_eb=eb_data['diff_eb'],
        **nan_stats
    )


def data_compare(cpu_data, npu_data, rank):
    rst_npu = verify_tensor_result(npu_data, cpu_data, "{}_dq_npu".format(rank))
    return rst_npu.check_result(rst_npu)[0] == 'ok'


def verify_result(out_dir):
    for rank in range(1):
        cpu_out_path = f"{out_dir}/cpu_out_{rank}.bin"
        npu_out_path = f"{out_dir}/out_{rank}.bin"
        
        cpu_out = torch.tensor(np.fromfile(cpu_out_path, np.float32))
        npu_out = torch.tensor(np.fromfile(npu_out_path, np.float16).astype(np.float32))
        npu_out = npu_out.cpu().view(-1)

        if not data_compare(cpu_out, npu_out, rank):
            print(f"============ out_{rank} precision check failed")
            return False

    print("test pass")
    return True


if __name__ == '__main__':
    verify_result(sys.argv[1])