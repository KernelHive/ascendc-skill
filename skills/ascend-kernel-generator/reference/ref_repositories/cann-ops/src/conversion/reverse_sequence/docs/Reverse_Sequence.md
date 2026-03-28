声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# ReverseSequence

## 支持的产品型号

Atlas 推理系列产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：实现反转可变长度切片

  **说明：**
  无。

## 实现原理

  沿着batch_axis维度对输入进行切片，对每个切片i，会在seq_axis维度上反转前seq_lengrhs[i]个元素。

## 算子执行接口

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnReverseSequenceGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnReverseSequence”接口执行计算。

* `aclnnStatus aclnnReverseSequenceGetWorkspaceSize(const aclTensor *x, const aclTensor *seqLengths, int64_t seqDim, int64_t batchDim, const aclTensor *out, uint64_t workspaceSize, aclOpExecutor **executor);`
* `aclnnStatus aclnnReverseSequence(void *workspace, int64_t workspaceSize, aclOpExecutor **executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnReverseSequenceGetWorkspaceSize

- **参数说明：**
  
  - x（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，需要被反转的输入Tensor，数据类型支持FLOAT、FLOAT16、INT8、INT16、UINT16、UINT8、INT32、INT64、BOOL、DOUBLE、COMPLEX64、COMPLEX128，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - seq_lengths（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，1维Tensor，，数据类型支持INT32、INT64，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - seq_dim（aclAttr\*，计算输入）:必选参数，取反的维度数据，类型支持INT，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - batch_dim（aclAttr\*，计算输入）:可选参数，默认为‘0’，沿着的维度执行反转，类型支持INT，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - y（aclTensor\*，计算输出）：Device侧的aclTensor，反转后的输出，数据类型和shape与X相同，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**
  
  返回aclnnStatus状态码。

### aclnnReverseSequence

- **参数说明：**
  
  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnReverseSequenceGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**
  
  返回aclnnStatus状态码。

## 约束与限制

- 无。

## 算子原型

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">AddCustom</td></tr>
</tr>
<tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">3-4维</td><td align="center">FLOAT、FLOAT16、INT8、INT16、UINT16、UINT8、INT32、INT64、BOOL、DOUBLE、COMPLEX64、COMPLEX128</td><td align="center">ND</td></tr>
<tr><td align="center">seq_lengths</td><td align="center">1维</td><td align="center">INT32、INT64</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="3" align="center">属性</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">seq_dim</td><td align="center">0维</td><td align="center">INT32、INT64</td><td align="center">ND</td></tr>
<tr><td align="center">batch_dim(可选)</td><td align="center">0维</td><td align="center">INT32、INT64</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">3-4维</td><td align="center">FLOAT、FLOAT16、INT8、INT16、UINT16、UINT8、INT32、INT64、BOOL、DOUBLE、COMPLEX64、COMPLEX128</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">reverse_sequence</td></tr>
</table>

## 调用示例

详见[ReverseSequence自定义算子样例说明算子调用章节](../README.md#算子调用)
