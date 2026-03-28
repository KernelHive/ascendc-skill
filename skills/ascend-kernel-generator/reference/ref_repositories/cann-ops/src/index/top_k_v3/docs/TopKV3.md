# aclnnTopk

## 支持的产品型号

- Atlas 推理系列产品/Atlas 训练系列产品/Atlas A2 训练系列产品/Atlas A3 训练系列产品

## 接口原型
每个算子分为两段式接口，必须先调用“aclnnTopkGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnTopk”接口执行计算。

- `aclnnStatus aclnnTopkGetWorkspaceSize(const aclTensor *self, int64_t k, int64_t dim, bool largest, bool sorted, aclTensor *valuesOut, aclTensor *indicesOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnTopk(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`

## 功能描述

算子功能：返回输入Tensor在指定维度上的k个极值及索引。

## aclnnTopkGetWorkspaceSize

- **参数说明**

  - self（aclTensor\*, 计算输入）： Device侧的aclTensor。shape支持1-8维度，支持非连续的Tensor, 数据格式支持ND。
    - Atlas 推理系列产品/Atlas 训练系列产品：数据类型支持INT8、UINT8、INT16、INT32、INT64、FLOAT16、FLOAT32、DOUBLE。
    - Atlas A2 训练系列产品/Atlas A3 训练系列产品：数据类型支持INT8、UINT8、INT16、INT32、INT64、FLOAT16、FLOAT32、DOUBLE、BFLOAT16。
  - k（int64_t\*, 计算输入）：Host侧的整型。表示计算维度上输出的极值个数。取值范围为[0, self.size(dim)]。
  - dim（int64_t\*, 计算输入）：Host侧的整型。表示计算维度。取值范围为[-self.dim(), self.dim())。
  - largest（bool\*, 计算输入）：Host侧的布尔型。True表示计算维度上的结果应由大到小输出，False表示计算维度上的结果由小到大输出。
  - sorted（bool\*, 计算输入）：Host侧的布尔型。True表示输出结果排序（若largest为True则结果从大到小排序，否则结果从小到大排序），False表示输出结果不排序，按输入时的数据顺序输出。注意：当前该参数仅支持取True,暂不支持取False。
  - valuesOut（aclTensor\*, 计算输出）：Device侧的aclTensor，数据类型与self保持一致。支持非连续的Tensor, 数据格式支持ND。shape排序轴与k一致，非排序轴与self一致。
    - Atlas 推理系列产品/Atlas 训练系列产品：数据类型支持INT8、UINT8、INT16、INT32、INT64、FLOAT16、FLOAT32、DOUBLE。
    - Atlas A2 训练系列产品/Atlas A3 训练系列产品：数据类型支持INT8、UINT8、INT16、INT32、INT64、FLOAT16、FLOAT32、DOUBLE、BFLOAT16。
  - indicesOut（aclTensor\*, 计算输出）：Device侧的aclTensor，数据类型支持INT64。支持非连续的Tensor, 数据格式支持ND。shape排序轴与k一致，非排序轴与self一致。
  - workspaceSize（uint64_t\*, 出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\**, 出参）：返回op执行器，包含了算子计算流程。

- **返回值**

  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）: 1. 传入的self、valuesOut或indicesOut是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）: 1. self、valuesOut或indicesOut的数据类型和数据格式不在支持的范围之内。
                                        2. dim不在输入self的合理维度范围内。
                                        3. k小于0或者k大于输入self在dim维度上的size大小。
  ```

## aclnnTopk

- **参数说明**

  - workspace（void\*, 入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t, 入参）：在Device侧申请的workspace大小，由第一段接口aclnnTopkGetWorkspaceSize获取。
  - executor（aclOpExecutor\*, 入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream, 入参）：指定执行任务的AscendCL Stream流。

- **返回值**

  aclnnStatus：返回状态码。

## 约束与限制

无。

## 调用示例

详见[TopKV3自定义算子样例说明算子调用章节](../README.md#算子调用)