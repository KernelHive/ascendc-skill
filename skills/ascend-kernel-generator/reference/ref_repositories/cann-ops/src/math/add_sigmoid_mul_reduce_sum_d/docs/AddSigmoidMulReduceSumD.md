声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# AddSigmoidMulReduceSumD

## 支持的产品型号

Atlas A2训练系列产品/Atlas 800I A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：实现了数据经过相加、相乘、sigmoid、相乘、以索引1合轴的计算，返回结果的功能。
- 计算公式：
  
  $$\text{add_res} = \text{add_0_input0} + \text{add_0_input1}$$
  $$\text{mul_1_res} = \text{add_res} \cdot \text{mul_0_input1}$$
  $$\text{sigmoid_res} = \frac{1}{1 + e^{-\text{mul_1_res}}}$$
  $$\text{mul_2_res} = \text{sigmoid_res} \cdot \text{mult_1_input1}$$
  $$\text{mul_3_res} = \text{mul_2_res} \cdot \text{mult_2_input1}$$
  $$\text{result} = \sum_{i} \text{mul_3_res}_i$$
  
  **说明：**
  无。

## 实现原理

AddSigmoidMulReduceSumD由add、mul、sigmoid、reduce_sum操作组成，计算过程为：

1. add_res = Add(add_0_input0, add_0_input1)
2. mul_1_res = Mul(add_res, mul_0_input1)
3. sigmoid_res = Sigmois(mul_1_res)
4. mul_2_res = Mul(sigmoid_res, mult_1_input1)
5. mul_3_res = Mul(mul_2_res, mult_2_input1)
6. result = ReduceSum(mul_3_res, axit=1)

## 算子执行接口

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnAddSigmoidMulReduceSumDGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnAddSigmoidMulReduceSumD”接口执行计算。

* `aclnnStatus aclnnAddSigmoidMulReduceSumDGetWorkspaceSize(const aclTensor *add_0_input0, const aclTensor *add_0_input1, const aclTensor *mul_0_input1, const aclTensor *mul_1_input1, const aclTensor *mul_2_input1, const aclTensor *out, uint64_t workspaceSize, aclOpExecutor **executor);`
* `aclnnStatus aclnnAddSigmoidMulReduceSumD(void *workspace, int64_t workspaceSize, aclOpExecutor **executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnAddSigmoidMulReduceSumDGetWorkspaceSize

- **参数说明：**
  
  - add_0_input0（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入add_0_input0，数据类型支持FLOAT16，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持NZ/NHWC/ND。
  - add_0_input1（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入add_0_input1，数据类型支持FLOAT16，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持NHWC/NHWC/ND。
  - mul_0_input1（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入mul_0_input1，数据类型支持FLOAT16，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持NHWC/NHWC/ND。
  - mul_1_input1（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入mul_1_input1，数据类型支持FLOAT16，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持NHWC/NHWC/ND。
  - mul_2_input1（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入mul_2_input1，数据类型支持FLOAT16，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持NZ/NHWC/ND。
  - out（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出out，数据类型支持FLOAT16，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持NZ/NHWC/ND。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

### aclnnAddSigmoidMulReduceSumD

- **参数说明：**
  
  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnAddSigmoidMulReduceSumDGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束与限制

- add_0_input0，add_0_input1，mul_0_input1, mul_1_input1, mul_2_input1, out的数据类型只支持FLOAT16

## 算子原型

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">AddSigmoidMulReduceSumD</td></tr>
</tr>
<tr><td rowspan="6" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">add_0_input0</td><td align="center">40, 4, 16, 2, 16(NZ)/40, 64, 32(ND/NHWC)</td><td align="center">float16</td><td align="center">NZ/NHWC/ND</td></tr>
<tr><td align="center">add_0_input1</td><td align="center">1, 1, 32</td><td align="center">float16</td><td align="center">NHWC/NHWC/ND</td></tr>
<tr><td align="center">mul_0_input1</td><td align="center">1</td><td align="center">float16</td><td align="center">NHWC/NHWC/ND</td></tr>
<tr><td align="center">mul_1_input1</td><td align="center">40, 64, 1</td><td align="center">float16</td><td align="center">NHWC/NHWC/ND</td></tr>
<tr><td align="center">mul_2_input1</td><td align="center">40, 4, 16, 2, 16(NZ)/40, 64, 32(ND/NHWC)</td><td align="center">float16</td><td align="center">NZ/NHWC/ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">out</td><td align="center">40, 4, 16, 2, 16(NZ)/40, 64, 32(ND/NHWC)</td><td align="center">float16</td><td align="center">NZ/NHWC/ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">add_sigmoid_mul_reduce_sum_d</td></tr>
</table>

## 调用示例

详见[AddSigmoidMulReduceSumD自定义算子样例说明算子调用章节](../README.md#算子调用)
