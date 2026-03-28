# GemmV2

## 支持的产品型号

- 昇腾910B AI处理器。

## 功能描述

- 计算α乘以A与B的乘积，再与β和input C的乘积求和。
- 计算公式：

  $$
  out=α(A @ B) + βC
  $$

## 算子原型

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">GemmV2</td></tr>
</tr>
<tr><td rowspan="6" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">A</td><td align="center">2 * 2</td><td align="center">float16、bfloat16</td><td align="center">ND</td></tr>
<tr><td align="center">B</td><td align="center">2 * 2</td><td align="center">float16、bfloat16</td><td align="center">ND</td></tr>
<tr><td align="center">alpha</td><td align="center">1</td><td align="center">float16、bfloat16</td><td align="center">ND</td></tr>
<tr><td align="center">beta</td><td align="center">1</td><td align="center">float16、bfloat16</td><td align="center">ND</td></tr>
<tr><td align="center">C</td><td align="center">2 * 2</td><td align="center">float32</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">out</td><td align="center">2 * 2</td><td align="center">float32</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">gemm_v2</td></tr>
</table>

## 调用示例

详见[GemmV2自定义算子样例说明算子调用章节](../README.md#算子调用)

