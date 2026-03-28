# aclnnForeachPowScalar

## 支持的产品型号

- Atlas A2 训练系列产品。

## 接口原型

每个算子分为两段式接口，必须先调用“aclnnForeachPowScalarGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnForeachPowScalar”接口执行计算。

- `aclnnStatus aclnnForeachPowScalarGetWorkspaceSize(const aclTensorList *x, const aclTensor *scalar, aclTensorList *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnForeachPowScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- 算子功能：返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入张量列表的每个张量进行n次方运算的结果。

- 计算公式：

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  y_i = {{x_i}}^{exponent} (i=0,1,...n-1)
  $$

## aclnnForeachPowScalarGetWorkspaceSize

- **参数说明**：

  - x（aclTensorList*，计算输入）：公式中的`x`，Device侧的aclTensorList，表示进行n次方运算的底数。数据类型支持FLOAT、FLOAT16、BFLOAT16、INT32。数据格式支持ND，shape维度不高于8维。支持非连续的Tensor，不支持空Tensor。
  - scalar（aclTensor*，计算输入）：公式中的`exponent`，Host侧的aclTensor，表示进行n次方运算的指数。数据格式支持ND。支持非连续的Tensor，不支持空Tensor。
  数据类型支持FLOAT、FLOAT16、INT32，且与入参`x`的数据类型具有一定对应关系：
    - 当`x`的数据类型为FLOAT、FLOAT16、INT32时，数据类型与`x`的数据类型保持一致。
    - 当`x`的数据类型为BFLOAT16时，数据类型支持FLOAT。
  - out（aclTensorList*，计算输出）：公式中的`y`，Device侧的aclTensorList，表示进行n次方运算的输出结果。数据类型支持FLOAT、FLOAT16、BFLOAT16、INT32，数据格式支持ND，shape维度不高于8维。数据类型、数据格式和shape跟入参`x`的数据类型、数据格式和shape一致。支持非连续的Tensor，不支持空Tensor。
  - workspaceSize（uint64_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\**，出参）：返回op执行器，包含了算子计算流程。

- **返回值**：

  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的x、scalar、out是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. x、scalar和out的数据类型不在支持的范围之内。
                                        2. x、scalar和out无法做数据类型推导。
  ```

## aclnnForeachPowScalar

- **参数说明**：

  - workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnForeachPowScalarGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。

- **返回值**：

  aclnnStatus：返回状态码。

## 约束与限制

无。
