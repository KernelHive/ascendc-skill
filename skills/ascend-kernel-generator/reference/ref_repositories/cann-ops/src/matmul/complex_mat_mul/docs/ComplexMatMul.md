声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# ComplexMatMul

## 支持的产品型号

Atlas A2 训练系列产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：ComplexMatMul算子实现复数矩阵乘法功能，支持批量矩阵乘法和偏置加法。该算子在信号处理、量子计算等领域有广泛应用。
- 计算公式：
  $$
  \text{output} = \text{matmul}(x, y) + \text{bias}
  $$
  其中，x, y, bias均为复数（complex64）张量，bias为可选参数。

## 实现原理

调用`Ascend C`的`API`接口实现复数矩阵乘法。对于批量矩阵乘法，通过循环处理每个矩阵对。支持在矩阵乘法后添加偏置。

## 算子执行接口

每个算子分为两段式接口，必须先调用“aclnnComplexMatMulGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnComplexMatMul”接口执行计算。

* `aclnnStatus aclnnComplexMatMulGetWorkspaceSize(const aclTensor* x, const aclTensor* y, const aclTensor* bias, const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnStatus aclnnComplexMatMul(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnComplexMatMulGetWorkspaceSize

- **参数说明：**

  - x（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，第一个输入矩阵，数据类型为complex64，数据格式支持ND。
  - y（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，第二个输入矩阵，数据类型为complex64，数据格式支持ND。
  - bias（aclTensor\*，计算输入）：可选参数，Device侧的aclTensor，偏置矩阵，数据类型为complex64，数据格式支持ND。
  - out（aclTensor\*，计算输出）：Device侧的aclTensor，输出矩阵，数据类型为complex64，数据格式支持ND。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数x, y, out是空指针，则返回161001。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：x, y, bias, out的数据类型和数据格式不在支持的范围内。
  ```

### aclnnComplexMatMul

- **参数说明：**

  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnComplexMatMulGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。


## 约束与限制

- x, y, bias, out的数据类型为complex64，数据格式只支持ND
- 输入矩阵x和y的维度必须满足矩阵乘法的要求（即x的最后一维与y的倒数第二维相等）
- 目前只支持Atlas A2训练系列产品

## 算子原型

<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">ComplexMatMul</th></tr> 
<tr><td align="center"> </td><td align="center">name</td><td align="center">type</td><td align="center">data type</td><td align="center">format</td></tr>  
<tr><td rowspan="3" align="center">算子输入</td>
<td align="center">x</td><td align="center">tensor</td><td align="center">complex64</td><td align="center">ND</td></tr>
<td align="center">y</td><td align="center">tensor</td><td align="center">complex64</td><td align="center">ND</td></tr>
<td align="center">bias</td><td align="center">tensor</td><td align="center">complex64</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">算子输出</td>
<td align="center">out</td><td align="center">tensor</td><td align="center">complex64</td><td align="center">ND</td></tr>  
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">complex_mat_mul</td></tr>  
</table>

## 调用示例

详见[ComplexMatMul自定义算子样例说明算子调用章节](../README.md#算子调用)
