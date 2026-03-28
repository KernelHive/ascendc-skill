# Neg

## 支持的产品型号

- Atlas A2训练系列产品
- Atlas 800I A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：Neg算子提供数值取负运算功能，对输入的数值型数据逐元素进行取负操作。
- 计算公式：

  $$
  y=−x
  $$

## 实现原理

输入的`int32, float16, float32`类型数据在kernel侧以原始数据类型进行处理，对于`int8`类型输入通过调用`Ascend C`的`Cast`函数将输入的`int8`数据转换为`float16`后进行计算，之后再次通过`Cast`函数将`float16`数据转为`int16`处理溢出，最后将结果转为`int8`返回，对于`bfloat16`数据类型将其通过Cast接口转换为32位浮点数进行计算， 实现对输入的数值型数据逐元素进行取负操作。

## 算子执行接口

每个算子分为两段式接口，必须先调用“aclnnNegGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnNeg”接口执行计算。

* `aclnnStatus aclnnNegGetWorkspaceSize(const aclTensor* x, const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnStatus aclnnNeg(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnNegGetWorkspaceSize

- **参数说明：**

  - x（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入x，数据类型支持float16、float32、int32、int8，数据格式支持ND。
  - out（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出y，数据类型与x一致，数据格式支持ND，输出维度与x一致。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：x、out的数据类型和数据格式不在支持的范围内。
  ```

### aclnnNeg

- **参数说明：**

  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnNegGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。


## 约束与限制

- x，y的数据类型支持INT32, INT8, FLOAT16, BFLOAT16, FLOAT32，数据格式只支持ND

## 算子原型

<table> 
<tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">Neg</th></tr> 
<tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr> <tr><td rowspan="1" align="center">算子输入</td> 
<td align="center">x</td><td align="center">tensor</td> <td align="center">int32, int8, float16, bfloat16, float32</td><td align="center">ND</td></tr> 

<tr><td rowspan="1" align="center">算子输出</td> 
<td align="center">y</td><td align="center">tensor</td> <td align="center">与输入相同</td><td align="center">ND</td></tr> 
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">neg</td></tr> 
</table>

## 调用示例

详见[Neg自定义算子样例说明算子调用章节](../README.md#算子调用)