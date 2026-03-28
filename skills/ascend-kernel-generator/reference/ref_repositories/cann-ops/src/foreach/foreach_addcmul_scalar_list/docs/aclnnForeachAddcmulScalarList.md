# aclnnForeachAddcmulScalarList

## 支持的产品型号

- Atlas A2 训练系列产品。

## 接口原型

每个算子分为两段式接口，必须先调用“aclnnForeachAddcmulScalarListGetWorkspaceSize” 接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnForeachAddcmulScalarList”接口执行计算。

- `aclnnStatus aclnnForeachAddcmulScalarListGetWorkspaceSize(const aclTensorList *x1, const aclTensorList *x2, const aclTensorList *x3, const aclTensor *scalars, aclTensorList *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnForeachAddcmulScalarList(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- 算子功能：
  
  返回一个和输入张量列表同样形状大小的新张量列表，对张量列表x2和张量列表x3执行逐元素乘法，将结果与标量列表scalars进行逐元素乘法，最后将结果与张量列表x1执行逐元素加法。

- 计算公式：
  
  $$
  x1 = [{x1_0}, {x1_1}, ... {x1_{n-1}}], x2 = [{x2_0}, {x2_1}, ... {x2_{n-1}}], x3 = [{x3_0}, {x3_1}, ... {x3_{n-1}}]\\
  scalars = [{scalars_0}, {scalars_1}, ... {scalars_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  {\rm y}_i = x1_{i} + {\rm scalars}_i × x2_{i} × x3_{i} (i=0,1,...n-1)
  $$

## aclnnForeachAddcmulScalarListGetWorkspaceSize

- **参数说明**：

  - x1（aclTensorList*，计算输入）：公式中的`x1`，Device侧的aclTensorList，表示进行混合运算中加法的第一个输入张量列表。数据类型支持FLOAT、FLOAT16、BFLOAT16、INT32。shape维度不高于8维，数据格式支持ND。shape与入参`x2`、`x3`、`scalars`和出参`out`的shape一致。支持非连续的Tensor，不支持空Tensor。该参数中所有tensor的数据类型保存一致。
  - x2（aclTensorList*，计算输入）：公式中的`x2`，Device侧的aclTensorList，表示进行混合运算中乘法的第二个输入张量列表。数据类型支持FLOAT、FLOAT16、BFLOAT16、INT32。shape维度不高于8维，数据格式支持ND。数据类型、数据格式和shape跟入参`x1`一致。支持非连续的Tensor，不支持空Tensor。该参数中所有tensor的数据类型保存一致。
  - x3（aclTensorList*，计算输入）：公式中的`x3`，Device侧的aclTensorList，表示进行混合运算中乘法的第三个输入张量列表。数据类型支持FLOAT、FLOAT16、BFLOAT16、INT32。shape维度不高于8维，数据格式支持ND。数据类型、数据格式和shape跟入参`x1`一致。支持非连续的Tensor，不支持空Tensor。该参数中所有tensor的数据类型保存一致。
  - scalars（aclTensor*，计算输入）：公式中的`scalars`，Host侧的aclTensor。表示进行混合运算中乘法的第一个输入标量列表。数据类型支持FLOAT、FLOAT16、BFLOAT16、INT32。支持维度不高于8维。数据格式支持ND。数据类型和数据格式跟入参`x1`一致。支持非连续的Tensor，不支持空Tensor。
  - out（aclTensorList*，计算输出）：公式中的`y`，Device侧的aclTensorList，表示进行混合运算的输出张量列表。数据类型支持FLOAT、FLOAT16、BFLOAT16、INT32。shape维度不高于8维，数据格式支持ND。数据类型、数据格式和shape跟入参`x1`一致。支持非连续的Tensor，不支持空Tensor。该参数中所有tensor的数据类型保存一致。
  - workspaceSize（uint64_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\**，出参）：返回op执行器，包含了算子计算流程。

- **返回值**：

  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的x1、x2、x3、scalars和out是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. x1、x2、x3、scalars和out的数据类型不在支持的范围之内。
                                        2. x1、x2、x3、scalars和out无法做数据类型推导。
  返回561002（ACLNN_ERR_INNER_TILING_ERROR）:1. x1与out的数据类型或者shape不一致。
  										   2. x1、x2、x3或out中的tensor的元素数据类型不一致。
  										   3. x1、x2、x3或out中的tensor维度超过8维。
  										   4. x1与x2、x1与x3的数据类型或者shape不一致。
  ```

## aclnnForeachAddcmulScalarList

- **参数说明**：

  - workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnForeachAddcmulScalarListGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。

- **返回值**：

  aclnnStatus：返回状态码。

## 约束与限制

无。
