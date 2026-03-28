# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import numpy as np
import os
# 'pylint: disable=too-few-public-methods


class Constant:
    """
    The class for constant
    """
    # Taylor coefficient
    COEF_AN = (1.13681498971755972054E-11,
               8.49262267667473811108E-10,
               1.94434204175553054283E-8,
               9.53151741254484363489E-7,
               3.07828309874913200438E-6,
               3.52513368520288738649E-4,
               -8.50149846724410912031E-4,
               4.22618223005546594270E-2,
               -9.17480371773452345351E-2,
               9.99999999999999994612E-1)
    COEF_AN_COUNT = 9
    COEF_AD = (2.40372073066762605484E-11,
               1.48864681368493396752E-9,
               5.21265281010541664570E-8,
               1.27258478273186970203E-6,
               2.32490249820789513991E-5,
               3.25524741826057911661E-4,
               3.48805814657162590916E-3,
               2.79448531198828973716E-2,
               1.5887424E-1,
               5.74918629489320327824E-1,
               1.00000000000000000539E0)
    COEF_AD_COUNT = 10

    COEF_BN = (5.08955156417900903354E-1,
               -2.44754418142697847934E-1,
               9.41512335303534411857E-2,
               -2.18711255142039025206E-2,
               3.66207612329569181322E-3,
               -4.23209114460388756528E-4,
               3.59641304793896631888E-5,
               -2.14640351719968974225E-6,
               9.10010780076391431042E-8,
               -2.40274520828250956942E-9,
               3.59233385440928410398E-11)
    COEF_BN_COUNT = 10
    COEF_BD = (-6.31839869873368190192E-1,
               2.36706788228248691528E-1,
               -5.31806367003223277662E-2,
               8.48041718586295374409E-3,
               -9.47996768486665330168E-4,
               7.81025592944552338085E-5,
               -4.55875153252442634831E-6,
               1.89100358111421846170E-7,
               -4.91324691331920606875E-9,
               7.18466403235734541950E-11)
    COEF_BD_COUNT = 10

    COEF_CN = (-5.90592860534773254987E-1,
               6.29235242724368800674E-1,
               -1.72858975380388136411E-1,
               1.64837047825189632310E-2,
               -4.86827613020462700845E-4)
    COEF_CN_COUNT = 4
    COEF_CD = (-2.69820057197544900361E0,
               1.73270799045947845857E0,
               -3.93708582281939493482E-1,
               3.44278924041233391079E-2,
               -9.73655226040941223894E-4)
    COEF_CD_COUNT = 5

    THRESHOLD_3_25 = 3.25
    THRESHOLD_6_25 = 6.25
    THRESHOLD_1E_9 = 1.0e9


def _polevl(data_x, coef, n):
    """
    y = polevl( x, coef, n );
    DESCRIPTION:
    Evaluates polynomial of degree N:
                        2          N
    y  =  C  + C x + C x  +...+ C x
             0    1     2          N
    Coefficients are stored in reverse order:
    coef[0] = C  , ..., coef[N] = C  .
                 N                   0
    Parameters:
    ----------
    data_x : the placeholder of data input
    coef : coef of the data
    iter_n : number of the coef

    Returns : A Tensor. Has the same type as data.
    -------
    """
    res = coef[0]
    for index in range(1, n + 1):
        res = res * data_x
        res = res + coef[index]
    return res


def _p1evl(data_x, coef, n):
    """
    y = p1evl( x, coef, N );
    DESCRIPTION:
    Evaluates polynomial of degree N:
                        2          N
    y  =  C  + C x + C x  +...+ C x
             0    1     2          N
    Coefficients are stored in reverse order:
    coef[0] = C  , ..., coef[N] = C  .
                 N                   0
    The function p1evl() assumes that coef[N] = 1.0 and is
    omitted from the array.  Its calling arguments are
    otherwise the same as polevl().
    -------
    """
    res = coef[0]
    res = res + data_x
    for index in range(1, n):
        res = res * data_x
        res = res + coef[index]
    return res


def _calc_condition_lt_three_p_two_five(input_x):
    """
    do arcsinx compute use the 15th order taylor expansion when 0 <= x < 3.25
    x = xx*xx
    y = xx * polevl( x, AN, 9 )/polevl( x, AD, 10 )

    Parameters:
    ----------
    input_x : the data input
    -------
    """
    data_square = input_x * input_x
    data_polevl_an = _polevl(data_square, Constant.COEF_AN, Constant.COEF_AN_COUNT)
    data_polevl_ad = _polevl(data_square, Constant.COEF_AD, Constant.COEF_AD_COUNT)
    res = input_x * data_polevl_an
    res = res / data_polevl_ad

    return res


def _calc_condition_lt_six_p_two_five(input_x):
    """
    do arcsinx compute use the 15th order taylor expansion when 3.25 <= x < 6.25
    x = 1.0/(xx*xx);
    y = (1.0/xx + x * polevl( x, BN, 10) / (p1evl( x, BD, 10) * xx)) * 0.5

    Parameters:
    ----------
    input_x : the data input
    -------
    """
    data_temp = input_x * input_x
    data_temp = 1 / data_temp
    data_rec = 1 / input_x
    data_polevl_bn = _polevl(data_temp, Constant.COEF_BN, Constant.COEF_BN_COUNT)
    data_polevl_bn = data_polevl_bn * data_temp
    data_plevl_bd = _p1evl(data_temp, Constant.COEF_BD, Constant.COEF_BD_COUNT)
    data_plevl_bd = data_plevl_bd * input_x
    res = data_polevl_bn / data_plevl_bd
    res = data_rec + res
    res = res * 0.5

    return res


def _calc_condition_le_one_e_nine(input_x):
    """
    do arcsinx compute use the 15th order taylor expansion when 6.25 <= x <= 1.0e9
    x = 1.0/(xx*xx);
    y = (1.0/xx + x * polevl( x, CN, 4) / (p1evl( x, CD, 5) * xx)) * 0.5

    Parameters:
    ----------
    input_x : the data input
    -------
    """
    data_temp = input_x * input_x
    data_temp = 1 / data_temp
    data_rec = 1 / input_x
    data_polevl_cn = _polevl(data_temp, Constant.COEF_CN, Constant.COEF_CN_COUNT)
    data_polevl_cn = data_polevl_cn * data_temp
    data_plevl_cd = _p1evl(data_temp, Constant.COEF_CD, Constant.COEF_CD_COUNT)
    data_plevl_cd = data_plevl_cd * input_x
    res = data_polevl_cn / data_plevl_cd
    res = data_rec + res
    res = res * 0.5

    return res


def _calc_condition_gt_one_e_nine(input_x):
    """
    do arcsinx compute use the 15th order taylor expansion when x > 1.0e9
    y = 1/xx * 0.5

    Parameters:
    ----------
    input_x : the data input
    -------
    """
    res = 0.5 / input_x

    return res


def dawsn(x):
    input_x = x
    for i in np.nditer(input_x, op_flags=['readwrite']):
        sign = 1
        if i < 0:
            i[...] = -i
            sign = -1
        if i < 3.25:
            i[...] = _calc_condition_lt_three_p_two_five(i) * sign
        elif i < 6.25:
            i[...] = _calc_condition_lt_six_p_two_five(i) * sign
        elif i <= 1.0e9:
            i[...] = _calc_condition_le_one_e_nine(i) * sign
        else:
            i[...] = _calc_condition_gt_one_e_nine(i) * sign
    return [input_x, ]

def gen_golden_data_simple():
    test_type = np.float16
    shape = [32]
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x = np.random.uniform(-1, 1, shape).astype(test_type)
    input_x.tofile("./input/input_x.bin")
    golden = np.array(dawsn(input_x)).astype(test_type).reshape(shape)
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
