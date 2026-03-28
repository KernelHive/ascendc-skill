声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# AddCustom

## 支持的产品型号

Atlas A2训练系列产品/Atlas 800I A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：实现Matmul矩阵乘, 加上偏置的功能。
- 计算公式：
  
  $$
  C = A * B + Bias
  $$
  
  **说明：**
    - A、B为源操作数，A为左矩阵，形状为\[M, K]；B为右矩阵，形状为\[K, N]。
    - C为目的操作数，存放矩阵乘结果的矩阵，形状为\[M, N]。
    - Bias为矩阵乘偏置，形状为\[N]。对A*B结果矩阵的每一行都采用该Bias进行偏置。

## 实现原理

本样例中实现的是[m, n, k]固定为[32, 32, 32]的Matmul算子，包含bias输入，并使用Ascend C基础Api实现。
- kernel实现  
  Matmul算子的数学表达式为：
  $$
  C = A * B + Bias
  $$
  其中A的形状为[32, 32], B的形状为[32, 32], C的形状为[32, 32], Bias的形状为[1, 32]。具体请参考[mmad_sparation_arch.cpp](../op_kernel/mmad.cpp)。

  **注：本样例只支持使用硬件分离架构的产品如Atlas A2训练系列产品/Atlas 800I A2推理产品，由于样例使用的基础API均为Cube核指令，本样例设置了Cube Only模式，只调用Cube核完成计算，代码如下：
  ```c++
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
  ```

## 算子执行接口

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnMmadGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMmad”接口执行计算。

* `aclnnStatus aclnnMmadGetWorkspaceSize(const aclTensor *a, const aclTensor *b, const aclTensor *bias, const aclTensor *c, uint64_t* workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnMmad(void *workspace, int64_t workspaceSize, aclOpExecutor **executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnMmadGetWorkspaceSize

- **参数说明：**
  
  - a（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入矩阵a，数据类型支持FLOAT16，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - b（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入矩阵b，数据类型支持FLOAT16，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - bias（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入向量bias，数据类型支持FLOAT，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - c（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出矩阵c，数据类型支持FLOAT，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。
  
  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：A、B、Bias、C的数据类型和数据格式不在支持的范围内。
  ```

### aclnnMmad

- **参数说明：**
  
  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnMmadGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束与限制

- A、B的数据类型只支持FLOAT16，Bias、C的数据类型只支持FLOAT，数据格式只支持ND。

## 算子原型

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Mmad</td></tr>
</tr>
<tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">A</td><td align="center">32 * 32</td><td align="center">float16</td><td align="center">ND</td></tr>
<tr><td align="center">B</td><td align="center">32 * 32</td><td align="center">float16</td><td align="center">ND</td></tr>
<tr><td align="center">Bias</td><td align="center">1 * 32</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">C</td><td align="center">32 * 32</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">mmad</td></tr>
</table>

## 调用示例

详见[Mmad自定义算子样例说明算子调用章节](../README.md#算子调用)
