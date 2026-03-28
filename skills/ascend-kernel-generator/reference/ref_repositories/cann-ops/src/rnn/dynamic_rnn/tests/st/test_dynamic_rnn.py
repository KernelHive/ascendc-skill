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

import numpy as np
import torch


def gen_rnn_cpu(x_data, w_data, bias_num, h_new, c_new, forget_bias, direction, cell_clip, seq_length, gate_order):

    def sigmoid(x):
        s = 1 / (1 + np.exp(-x))
        return s

    t_size = x_data.shape[0]
    batch = x_data.shape[1]
    hidden = w_data.shape[1] // 4
    src_type = c_new.dtype

    for var in range(t_size):
        if direction == "UNIDIRECTIONAL":
            x_new = np.concatenate((x_data[var], h_new), axis=1)
        else:
            x_new = np.concatenate((x_data[t_size - 1 - var], h_new), axis=1)
        if np.size(seq_length) != 0:
            if direction == "UNIDIRECTIONAL":
                seq = seq_length[var]
            else:
                seq = seq_length[t_size - 1 - var]

        res = torch.matmul(torch.from_numpy(x_new).to(torch.float), torch.from_numpy(w_data).to(torch.float))
        res = res.numpy()

        if bias_num is not None:
            bias_num = bias_num.astype("float32")
            res = res + bias_num

        if gate_order == "ijfo":
            res_i, res_j, res_f, res_o = np.split(res, 4, axis=1)
        else:
            res_i, res_f, res_j, res_o = np.split(res, 4, axis=1)

        res_f = res_f + forget_bias
        res_i = sigmoid(res_i)
        res_j = np.tanh(res_j)
        res_f = sigmoid(res_f)
        res_o = sigmoid(res_o)

        c_tmp1 = c_new * res_f
        c_tmp2 = res_j * res_i

        if np.size(seq_length) == 0:
            c_new = c_tmp1 + c_tmp2
        else:
            c1 = c_tmp1 + c_tmp2
            c_new = (c1 - c_new) * seq + c_new

        if cell_clip > 0:
            c_new = np.minimum(c_new, cell_clip)

        c_tmph = np.tanh(c_new)
        output_y = None
        if np.size(seq_length) == 0:
            h_new = c_tmph * res_o  # (b, hidden)
        else:
            h1 = c_tmph * res_o
            h_new = (h1 - h_new) * seq + h_new
            y_new = h1 * seq
            if var == 0:
                output_y = y_new
            elif direction == "UNIDIRECTIONAL":
                output_y = np.concatenate((output_y, y_new), axis=0)
            else:
                output_y = np.concatenate((y_new, output_y), axis=0)

        h_new = h_new.astype('float32')

        if var == 0:
            output_h = h_new
            output_c = c_new
            output_i = res_i
            output_j = res_j
            output_f = res_f
            output_o = res_o
            output_tanc = c_tmph

        elif direction == "UNIDIRECTIONAL":
            output_h = np.concatenate((output_h, h_new), axis=0)
            output_c = np.concatenate((output_c, c_new), axis=0)
            output_i = np.concatenate((output_i, res_i), axis=0)
            output_j = np.concatenate((output_j, res_j), axis=0)
            output_f = np.concatenate((output_f, res_f), axis=0)
            output_o = np.concatenate((output_o, res_o), axis=0)
            output_tanc = np.concatenate((output_tanc, c_tmph), axis=0)
        else:
            output_h = np.concatenate((h_new, output_h), axis=0)
            output_c = np.concatenate((c_new, output_c), axis=0)
            output_i = np.concatenate((res_i, output_i), axis=0)
            output_j = np.concatenate((res_j, output_j), axis=0)
            output_f = np.concatenate((res_f, output_f), axis=0)
            output_o = np.concatenate((res_o, output_o), axis=0)
            output_tanc = np.concatenate((c_tmph, output_tanc), axis=0)

    output_h = output_h.reshape(t_size, batch, hidden).astype(src_type)
    output_c = output_c.reshape(t_size, batch, hidden).astype(src_type)
    output_i = output_i.reshape(t_size, batch, hidden).astype(src_type)
    output_j = output_j.reshape(t_size, batch, hidden).astype(src_type)
    output_f = output_f.reshape(t_size, batch, hidden).astype(src_type)
    output_o = output_o.reshape(t_size, batch, hidden).astype(src_type)
    output_tanc = output_tanc.reshape(t_size, batch, hidden).astype(src_type)

    if np.size(seq_length) == 0:
        output_y = output_h

    return [output_y, output_h, output_c, output_i, output_j, output_f, output_o, output_tanc]


def calc_expect_func(**kwargs):
    x = kwargs.get('x', {'value': None})['value']
    w = kwargs.get('w', {'value': None})['value']
    bias_num = kwargs.get('b', {'value': None})['value']
    init_h = kwargs.get('init_h', {'value': None})['value']
    init_c = kwargs.get('init_c', {'value': None})['value']

    forget_bias = kwargs.get('forget_bias', 0.0)
    gate_order = kwargs.get('gate_order', "ijfo")
    direction = kwargs.get('direction', "UNIDIRECTIONAL")
    cell_clip = kwargs.get('cell_clip', -1.0)
    seq_length = kwargs.get('seq_length', None)
    is_seq_length = kwargs.get('is_seq_length', False)

    init_h = init_h.squeeze(0)
    init_c = init_c.squeeze(0)

    if not is_seq_length:
        seq_length = np.array([])
    output_data = gen_rnn_cpu(x, w, bias_num, init_h, init_c, forget_bias,
                              direction, cell_clip, seq_length, gate_order)

    return output_data
