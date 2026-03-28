# aclnnForeachLerpScalar

## 支持的产品型号

- Atlas A2 训练系列产品。

## 接口原型

每个算子分为两段式接口，必须先调用“aclnnForeachLerpScalarGetWorkspaceSize” 接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnForeachLerpScalar”接口执行计算。

- `aclnnStatus aclnnForeachLerpScalarGetWorkspaceSize(const aclTensorList *x1, const aclTensorList *x2, const aclScalar *weight, aclTensorList *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnForeachLerpScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- 算子功能：
  
  返回一个和输入张量列表同样形状大小的新张量列表，其中每个元素都是张量列表x1和张量列表x2对应位置上元素的线性插值结果，其中标量值weight是插值系数。

- 计算公式：

  $$
  x1 = [{x1_0}, {x1_1}, ... {x1_{n-1}}], x2 = [{x2_0}, {x2_1}, ... {x2_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  {\rm y}_i = x1_i + {\rm weight} × (x2_i - x1_i) (i=0,1,...n-1)
  $$

## aclnnForeachLerpScalarGetWorkspaceSize

- **参数说明**：

  - x1（aclTensorList*，计算输入）：公式中的`x1`，Device侧的aclTensorList，表示进行线性插值计算的第一个输入张量列表。数据类型支持FLOAT、FLOAT16、BFLOAT16。shape维度不高于8维，数据格式支持ND。shape与入参`x2`和出参`out`的shape一致。支持非连续的Tensor，不支持空Tensor。
  - x2（aclTensorList*，计算输入）：公式中的`x2`，Device侧的aclTensorList，表示进行线性插值计算的第二个输入张量列表。数据类型支持FLOAT、FLOAT16、BFLOAT16。shape维度不高于8维，数据格式支持ND。数据类型、数据格式和shape跟入参`x1`的数据类型、数据格式和shape一致。支持非连续的Tensor，不支持空Tensor。
  - weight（aclScalar*，计算输入）：公式中的`weight`，Host侧的aclScalar，表示进行线性插值计算的插值系数。数据类型支持FLOAT。数据格式支持ND。
  - out（aclTensorList*，计算输出）：公式中的`y`，Device侧的aclTensorList，表示进行线性插值计算的输出张量列表。数据类型支持FLOAT、FLOAT16、BFLOAT16。shape维度不高于8维，数据格式支持ND。数据类型、数据格式和shape跟入参`x1`的数据类型、数据格式和shape一致。支持非连续的Tensor，不支持空Tensor。
  - workspaceSize（uint64_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\**，出参）：返回op执行器，包含了算子计算流程。

- **返回值**：

  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的x1、x2和out是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. x1、x2、weight和out的数据类型不在支持的范围之内。
                                        2. x1、x2、weight和out无法做数据类型推导。
  ```

## aclnnForeachLerpScalar

- **参数说明**：

  - workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnForeachLerpScalarGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。

- **返回值**：

  aclnnStatus：返回状态码。

## 约束与限制

无。
