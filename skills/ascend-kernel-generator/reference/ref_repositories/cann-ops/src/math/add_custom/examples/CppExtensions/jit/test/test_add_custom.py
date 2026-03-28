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

import os
import shutil
import unittest
import torch
import torch.utils.cpp_extension
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))


def remove_build_path():
    default_build_root = torch.utils.cpp_extension.get_default_build_root()
    if os.path.exists(default_build_root):
        shutil.rmtree(default_build_root)


class TestCppExtensionsJIT(TestCase):

    def set_up(self):
        super().set_up()
        self.old_working_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

    def tear_down(self):
        super().tear_down()
        os.chdir(self.old_working_dir)

    def _test_jit_compile_extension_with_cpp(self):
        extra_ldflags = []
        extra_ldflags.append("-ltorch_npu")
        extra_ldflags.append(f"-L{PYTORCH_NPU_INSTALL_PATH}/lib")
        extra_include_paths = []
        extra_include_paths.append("./")
        extra_include_paths.append(os.path.join(PYTORCH_NPU_INSTALL_PATH, "include"))
        extra_include_paths.append(os.path.join(PYTORCH_NPU_INSTALL_PATH, "include/third_party/acl/inc"))

        module = torch.utils.cpp_extension.load(name="jit_extension",
                                                sources=["../extension_add.cpp"],
                                                extra_include_paths=extra_include_paths,
                                                extra_cflags=["-g"],
                                                extra_ldflags=extra_ldflags,
                                                verbose=True)
        length = [8, 2048]
        x = torch.rand(length, device='cpu', dtype=torch.float16)
        y = torch.rand(length, device='cpu', dtype=torch.float16)
        res = module.add_custom(x.npu(), y.npu())

        self.assertRtolEqual(res.npu(), (x + y))

    def test_jit_compile_extension_with_cpp(self):
        self._test_jit_compile_extension_with_cpp()

    @classmethod
    def set_up_class(cls):
        super().set_up_class()
        remove_build_path()

    @classmethod
    def tear_down_class(cls):
        remove_build_path()


if __name__ == '__main__':
    run_tests()
