声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# Exp

## 支持的产品型号

- Atlas A2 训练系列产品
- Atlas 800I A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：该Exp 算子提供指数函数的计算功能。其主要功能是计算给定数值的自然指数幂，即(e^x\)。

- 在数学和工程领域中，自然指数运算具有广泛且关键的应用：

  - 在概率论与统计学中，常用于描述正态分布、指数分布等概率模型；
  - 在信号处理中，可用于求解微分方程、处理衰减或增长信号；
  - 在机器学习中，是 softmax 函数、激活函数（如指数线性单元 ELU）的核心组成部分；
  - 在物理模拟中，可用于刻画放射性衰变、种群增长等自然过程。
  - Exp 算子能够高效处理批量数值的自然指数计算，支持整数、浮点数等多种数值类型的输入，且在实现中通常会针对边界情况（如极大或极小输入值）进行优化，以保证计算的准确性和稳定性。

- 计算公式：

  $$
  y = e^x
  $$

## 实现原理

调用`Ascend C`的`API`接口Exp、`Muls`和Adds进行实现。对于16位的数据类型将其通过`Cast`接口转换为32位浮点数进行计算。


## 算子执行接口

每个算子分为两段式接口，必须先调用“aclnnExpGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnExp”接口执行计算。

* `aclnnStatus aclnnExpGetWorkspaceSize(const aclTensor *x, const aclTensor *out, uint64_t workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnExp(void *workspace, int64_t workspaceSize, aclOpExecutor **executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnExpGetWorkspaceSize

- **参数说明：**

  - x（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入x，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND,NCHW,NHWC。
  - y（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出y，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND,NCHW,NHWC，输出维度与x一致。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：x、out的数据类型和数据格式不在支持的范围内。
  ```

### aclnnExp

- **参数说明：**

  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnRsqrtGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束与限制

- x，y的数据类型支持FLOAT16，FLOAT32，BFLOAT16，数据格式只支持ND

## 

详见[Reciprocal自定义算子样例说明算子调用章节](../README.md#算子调用)