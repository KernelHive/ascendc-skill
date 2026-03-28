声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# KlDivTargetBackward

## 支持的产品型号

Atlas A2训练系列产品/Atlas 800I A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：实现了计算KL散度在反向传播过程中目标张量的梯度的功能，返回目标张量的梯度。
- 计算公式：
  
  $$tmp = \left\{ \begin{array}{r}
  gradOutput*\left\lbrack (target + 1 - self)*e^{target} \right\rbrack\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ if\ \ \ \ \ \ logTarget = true \\
  \begin{matrix}
  gradOutput*\left( \ln^{target} + 1 - self \right)\ \ \ \ \ \ \ \ if\ \ \ \ \ \ logTarget = false\ and\ target > 0 \\
  0\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ if\ \ \ \ \ \ logTarget = false\ and\ target = 0
  \end{matrix}
  \end{array} \right.\ $$

  $$gradTarget = \left\{ \begin{array}{r}
  \frac{tmp}{target.size(0)}\ \ \ \ \ \ \ \ \ \ if\ \ \ \ \ \ reduction = 1 \\
  tmp\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ else
  \end{array} \right.\ $$
  
  **说明：**
  gradOutput表示梯度输出，self是输入张量，target是目标张量，reduction指定KL散度正向计算完loss之后的操作，logTarget表示target的数据是否已经做过log操作, gradTarget是输出的target的梯度。

## 实现原理

调用Ascend C的API接口Adds、Sub、Exp、Mul、Ln、CompareScalar、Select、Muls实现。对于bfloat16的输入将其Cast成float进行计算。

## 算子执行接口

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnKlDivTargetBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnKlDivTargetBackward”接口执行计算。

* `aclnnStatus aclnnKlDivTargetBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, 
const aclTensor *target, int64_t reduction, bool logTarget, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnKlDivTargetBackward(void *workspace, int64_t workspaceSize, aclOpExecutor **executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnKlDivTargetBackwardGetWorkspaceSize

- **参数说明：**
  
  - gradOutput（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入gradOutput，表示输入张量的梯度，数据类型支持FLOAT16、FLOAT32、BFLOAT16，数据格式支持ND。
  - self（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入self，表示输入张量，数据类型支持FLOAT16、FLOAT32、BFLOAT16，数据格式支持ND。
  - target（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入target，表示目标张量，数据类型支持FLOAT16、FLOAT32、BFLOAT16，数据格式支持ND。
  - reduction（int64_t，入参）：指定KL散度正向计算完loss之后的操作，数据类型支持INT64。
  - logTarget（bool，入参）：target的数据是否已经做过log操作，数据类型支持BOOL。
  - gradTarget（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出gradTarget，表示目标张量的梯度，数据类型支持FLOAT16、FLOAT32、BFLOAT16，数据格式支持ND。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。
  
  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：输入和输出的数据类型和数据格式不在支持的范围内。
  ```

### aclnnKlDivTargetBackward

- **参数说明：**
  
  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnKlDivTargetBackwardGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束与限制

- 1. grad_output和target的shape需要与self满足broadcast关系；
- 2. reduction支持0（‘none’）| 1（‘mean’）| 2（‘sum’），‘none’表示不应用减少，‘mean’表示loss求均值，‘sum’表示loss求和

## 调用示例

详见[KlDivTargetBackward自定义算子样例说明算子调用章节](../README.md#算子调用)
