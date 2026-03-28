#!/usr/bin/python3
# coding=utf-8
#
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

torch.npu.config.allow_internal_format = False


class TestCustomKlDivTargetBackward(TestCase):
    def test_kl_div_target_backward(self):
        length = [8, 2048]
        grad_output = torch.rand(length, device='cpu', dtype=torch.float16)
        self_x = torch.rand(length, device='cpu', dtype=torch.float16)
        target = torch.rand(length, device='cpu', dtype=torch.float16)
        reduction = 0
        log_target = False
        if not log_target:
            tmp = torch.log(target)
            grad_target = tmp + 1
            grad_target = grad_target - self_x
            grad_target = grad_output * grad_target
            grad_target = grad_target.masked_fill(target == 0, 0)
        else:
            grad_target = target + 1
            grad_target = grad_target - self_x
            tmp = torch.exp(target)
            grad_target = grad_target * tmp
            grad_target = grad_output * grad_target

        if reduction == 1:
            max_len = max(max(grad_output.numel(), self_x.numel()), target.numel())
            grad_target = grad_target / max_len

        torch.npu.synchronize()
        output = torch_npu.npu_kl_div_target_backward(
            grad_output.npu(), self_x.npu(), target.npu(), reduction, log_target).cpu()
        torch.npu.synchronize()

        self.assertRtolEqual(output, grad_target)


if __name__ == "__main__":
    run_tests()
