# aclnnForeachCopy

## 支持的产品型号

- Atlas A2 训练系列产品。

## 接口原型

每个算子分为两段式接口，必须先调用“aclnnForeachCopyGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnForeachCopy”接口执行计算。

- `aclnnStatus aclnnForeachCopyGetWorkspaceSize(const aclTensorList *x, aclTensorList *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnForeachCopy(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- 算子功能：用于实现两个张量列表内容的复制，要求输入和输出两个张量列表形状相同。

- 计算公式：

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$ 

  $$
  {\rm y}_i = x_i (i=0,1,...n-1)
  $$

## aclnnForeachCopyGetWorkspaceSize

- **参数说明**：

  - x（aclTensorList*，计算输入）：公式中的`x`，Device侧的aclTensorList，表示进行内容复制的输入张量列表。数据类型支持FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、INT6、UINT16、INT32、UINT32、INT64、DOUBLE、BOOL。数据格式支持ND，shape维度不高于8维。
  - out（aclTensorList*，计算输出）：公式中的`y`，Device侧的aclTensorList，表示进行内容复制的输出张量列表。数据类型支持FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、INT6、UINT16、INT32、UINT32、INT64、DOUBLE、BOOL。数据格式支持ND，shape维度不高于8维。数据格式和shape跟入参`x`一致。
  - workspaceSize（uint64_t\*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\**，出参）：返回op执行器，包含了算子计算流程。

- **返回值**：

  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的x或out是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. x或out的数据类型不在支持的范围之内。
  ```

## aclnnForeachCopy

- **参数说明**：

  - workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnForeachCopyGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。

- **返回值**：

  aclnnStatus：返回状态码。

## 约束与限制

  无。
