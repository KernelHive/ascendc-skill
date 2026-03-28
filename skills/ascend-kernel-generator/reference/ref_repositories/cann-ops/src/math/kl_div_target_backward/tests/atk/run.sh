#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

atk case -f op_kl_div_target_backward.yaml -p generate_reduce.py
atk node --backend pyaclnn --devices 0 node --backend cpu task -c result/op_kl_div_target_backward/json/all_op_kl_div_target_backward.json -p aclnn_kl_div_target_backward.py --task accuracy -s 0 -e 703 -mt 100
atk node --backend pyaclnn --devices 0 node --backend npu --devices 1 task -c result/op_kl_div_target_backward/json/all_op_kl_div_target_backward.json -p aclnn_kl_div_target_backward.py --task performance_device -s 0 -e 703 -mt 100
atk node --backend pyaclnn --devices 0 node --backend cpu --is_compare False task -c result/op_kl_div_target_backward/json/all_op_kl_div_target_backward.json -p aclnn_kl_div_target_backward.py --task accuracy_dc -e 703 -rn 50

atk case -f op_kl_div_target_backward_broadcast.yaml -p generate_broadcast.py
atk node --backend pyaclnn --devices 0 node --backend cpu task -c result/op_kl_div_target_backward_broadcast/json/all_op_kl_div_target_backward_broadcast.json -p aclnn_kl_div_target_backward.py --task accuracy -s 0 -e 600 -mt 100
atk node --backend pyaclnn --devices 0 node --backend npu --devices 1 task -c result/op_kl_div_target_backward_broadcast/json/all_op_kl_div_target_backward_broadcast.json -p aclnn_kl_div_target_backward.py --task performance_device -s 0 -e 600 -mt 100
atk node --backend pyaclnn --devices 0 node --backend cpu --is_compare False task -c result/op_kl_div_target_backward_broadcast/json/all_op_kl_div_target_backward_broadcast.json -p aclnn_kl_div_target_backward.py --task accuracy_dc -e 703 -rn 50
