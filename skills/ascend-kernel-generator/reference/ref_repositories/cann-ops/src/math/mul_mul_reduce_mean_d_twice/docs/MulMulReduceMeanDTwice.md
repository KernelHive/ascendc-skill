声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# MulMulReduceMeanDTwice

## 支持的产品型号

Atlas A2训练系列产品/Atlas 800I A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：实现了数据相乘、第一次mean、平方差、第二次mean，返回计算结果的功能。
- 计算公式：
  
  $$
  \text{mul_res} = \text{mul0input0} \times \text{mul0input1} \times \text{mul1input0}
  $$
  $$
  \text{reduce_mean_0} = \frac{1}{N} \sum_{i=1}^{N} \text{mul_res}[i]
  $$
  $$
  \text{diff} = \text{mul_res} - \text{reduce_mean_0}
  $$
  $$
  \text{muld_res} = \text{diff} \times \text{diff}
  $$
  $$
  \text{x2} = \frac{1}{N} \sum_{i=1}^{N} \text{muld_res}[i]
  $$
  $$
  \text{reduce_mean_1} = \frac{\gamma}{\sqrt{\text{x2} + \text{addy}}}
  $$
  $$
  \text{output} = \beta - \text{reduce_mean_1} \times \text{reduce_mean_0} + \text{reduce_mean_1} \times \text{mul_res}
  $$

  $$
  \text{out} = \frac{1}{1 + e^{-(\text{output} \cdot \text{t}_1)}}
  $$
  
  **说明：**
  无。

## 实现原理

MulMulReduceMeanDTwice由Mul、第一次mean、平方差、第二次mean等操作组成，计算过程：

1. mul_res = mul0input0 * mul0input1 * mul1input0
2. reduce_mean_0 = np.mean(mul_res, axis=1, keepdims=True)
3. diff = mul_res - reduce_mean_0
4. muld_res = diff * diff
5. x2 = np.mean(muld_res, axis=1, keepdims=True)
6. reduce_mean_1 = gamma / np.sqrt(x2 + addy)
7. output = beta - reduce_mean_1 * reduce_mean_0 + reduce_mean_1 * mul_res

## 算子执行接口

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnMulMulReduceMeanDTwiceGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMulMulReduceMeanDTwice”接口执行计算。

* `aclnnStatus aclnnMulMulReduceMeanDTwiceGetWorkspaceSize(const aclTensor *mul0_input0, const aclTensor *mul0_input1, const aclTensor *mul1_input0, const aclTensor *add_y, const aclTensor *gamma, const aclTensor *beta, const aclTensor *output, uint64_t workspaceSize, aclOpExecutor **executor);`
* `aclnnStatus aclnnMulMulReduceMeanDTwice(void *workspace, int64_t workspaceSize, aclOpExecutor **executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnMulMulReduceMeanDTwiceGetWorkspaceSize

- **参数说明：**
  
  - mul0_input0（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入mul0_input0，数据类型支持FLOAT16/FLOAT32/BFLOAT16，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - mul0_input1（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入mul0_input1，数据类型支持FLOAT16/FLOAT32/BFLOAT16，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - mul1_input0（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入mul1_input0，数据类型支持FLOAT16/FLOAT32/BFLOAT16，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - add_y（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入add_y，数据类型支持FLOAT16/FLOAT32/BFLOAT16，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - gamma（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入gamma，数据类型支持FLOAT16/FLOAT32/BFLOAT16，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - beta（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入beta，数据类型支持FLOAT16/FLOAT32/BFLOAT16，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - output（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出outputs，数据类型支持FLOAT16/FLOAT32/BFLOAT16，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

### aclnnMulMulReduceMeanDTwiceGetWorkspaceSize

- **参数说明：**
  
  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnMulSigmoidMulAddCustomGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束与限制

- mul0_input0, mul0_input1, mul1_input0，add_y, gamma, beta，output的数据类型只支持FLOAT16/FLOAT32/BFLOAT16，数据格式只支持ND

## 算子原型

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">MulMulReduceMeanDTwice</td></tr>
</tr>
<tr><td rowspan="7" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">mul0_input0</td><td align="center">90 * 1024</td><td align="center">float16/float32/bfloat16</td><td align="center">ND</td></tr>
<tr><td align="center">mul0_input1</td><td align="center">90 * 1024</td><td align="center">float16/float32/bfloat16</td><td align="center">ND</td></tr>
<tr><td align="center">mul1_input0</td><td align="center">1</td><td align="center">float16/float32/bfloat16</td><td align="center">ND</td></tr>
<tr><td align="center">add_y</td><td align="center">1</td><td align="center">float16/float32/bfloat16</td><td align="center">ND</td></tr>
<tr><td align="center">gamma</td><td align="center">1 * 1024</td><td align="center">float16/float32/bfloat16</td><td align="center">ND</td></tr>
<tr><td align="center">beta</td><td align="center">1 * 1024</td><td align="center">float16/float32/bfloat16</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">output</td><td align="center">90 * 1024</td><td align="center">float16/float32/bfloat16</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">mul_mul_reduce_mean_d_twice</td></tr>
</table>

## 调用示例

详见[MulMulReduceMeanDTwice自定义算子样例说明算子调用章节](../README.md#算子调用)
