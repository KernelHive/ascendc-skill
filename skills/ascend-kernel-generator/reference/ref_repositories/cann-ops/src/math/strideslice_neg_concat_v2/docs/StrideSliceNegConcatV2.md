声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# StridesliceNegConcatV2

## 支持的产品型号

Atlas A2训练系列产品/Atlas 800I A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：实现了数据取反，返回结果的功能。
- 计算公式：
  
  $$
  {output} = \frac{1}{1 + e^{-(\text{input} \cdot \text{t})}}
  $$
  
  **说明：**
  无。

## 实现原理

StridesliceNegConcatV2由取反操作组成，计算过程：

1. N,H,W,C = input.shape
2. mid_col = C // 2
3. output = input.copy()
4. output[:,:,:,mid_col:] = -input[:,:,:,mid_col:]


## 算子执行接口

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnStridesliceNegConcatV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnStridesliceNegConcatV2”接口执行计算。

* `aclnnStatus aclnnStridesliceNegConcatV2GetWorkspaceSize(const aclTensor *input, const aclTensor *output, uint64_t workspaceSize, aclOpExecutor **executor);`
* `aclnnStatus aclnnStridesliceNegConcatV2(void *workspace, int64_t workspaceSize, aclOpExecutor **executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnStridesliceNegConcatV2GetWorkspaceSize

- **参数说明：**
  
  - input（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入input，数据类型支持FLOAT16/FLOAT32/BFLOAT16，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - output（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出output，数据类型支持FLOAT16/FLOAT32/BFLOAT16，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

### aclnnStridesliceNegConcatV2

- **参数说明：**
  
  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnStridesliceNegConcatV2GetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束与限制

- input, output的数据类型只支持FLOAT16/FLOAT32/BFLOAT16，数据格式只支持ND

## 算子原型

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">StridesliceNegConcatV2</td></tr>
</tr>
<tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">input</td><td align="center">1 * 128 * 1 * 128</td><td align="center">float16/float32/bfloat16</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">output</td><td align="center">1 * 128 * 1 * 128</td><td align="center">float16/float32/bfloat16</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">strideslice_neg_concat_v2</td></tr>
</table>

## 调用示例

详见[StridesliceNegConcatV2自定义算子样例说明算子调用章节](../README.md#算子调用)
