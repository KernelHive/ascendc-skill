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
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
gendata_dir = os.path.abspath(os.path.join(current_dir, '../../examples/AclNNInvocationNaive'))
if gendata_dir not in sys.path:
    sys.path.insert(0, gendata_dir)
from gen_data import radius_numpy, RadiusParams


def generate_case(case_id, compute_dtype=np.float32):
    x = np.random.uniform(-5, 5, [100, 2]).astype(compute_dtype)
    y = np.random.uniform(-5, 5, [50, 2]).astype(compute_dtype)
    ptr_x = None
    ptr_y = None
    r = 1.0
    max_num_neighbors = 10
    ignore_same_index = False
    assign_index = radius_numpy(x, y, RadiusParams(r, max_num_neighbors, ignore_same_index), ptr_x, ptr_y)
    if ptr_x is None:
        return {"input_desc": {"x": {"shape": list(x.shape), "value": x.tolist()}, 
                        "y": {"shape": list(y.shape), "value": y.tolist()},
                        },
            "attr": {"r": {"value": r}, "max_num_neighbors": {"value": max_num_neighbors},
                    "ignore_same_index": {"value": ignore_same_index}},
            "output_desc": {"out": {"shape": list(assign_index.shape)}}
            }
    else:
        return {"input_desc": {"x": {"shape": list(x.shape), "value": x.tolist()}, 
                        "y": {"shape": list(y.shape), "value": y.tolist()},
                        "ptr_x": {"shape": list(ptr_x.shape), "value": ptr_x.tolist()},
                        "ptr_y": {"shape": list(ptr_y.shape), "value": ptr_y.tolist()},
                        },
                            
            "attr": {"r": {"value": r}, "max_num_neighbors": {"value": max_num_neighbors},
                    "ignore_same_index": {"value": ignore_same_index}},
            "output_desc": {"out": {"shape": list(assign_index.shape)}}
            }


def fuzz_branch_001():
    return generate_case(1)
