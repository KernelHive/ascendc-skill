# aclnnComplexMatDot

## 支持的产品型号
- Atlas A2 训练系列产品/Atlas 800I A2 推理产品。

## 接口原型
每个算子分为两段式接口，必须先调用“aclnnComplexMatDotGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnComplexMatDot”接口执行计算。

- `aclnnStatus aclnnComplexMatDotGetWorkspaceSize(const aclTensor *matx, const aclTensor *maty, int64_t m, int64_t n, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnComplexMatDot(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述
- 算子功能：对两个复数矩阵按对应位置作逐点乘，返回结果与输入矩阵形状大小相同。  
- 计算公式：  
  $$
  {out_{ij}} = {matx_{ij}} * {maty_{ij}}
  $$
  其中，下标i、j表示第i行、第j列。

## aclnnComplexMatDotGetWorkspaceSize
- **参数说明**：
  
  - matx（aclTensor*，计算输入）：公式中的matx，Device侧的aclTensor，Atlas A2 训练系列产品/Atlas 800I A2推理产品数据类型支持COMPLEX64，shape维度不高于3维，数据格式支持ND。不支持非连续的Tensor，不支持空Tensor。
  - maty（aclTensor*，计算输入）：公式中的maty，Device侧的aclTensor，Atlas A2 训练系列产品/Atlas 800I A2推理产品数据类型支持COMPLEX64，数据格式支持ND。不支持非连续的Tensor，不支持空Tensor。
  - m（int64_t，入参）：表示矩阵行数，数据类型支持INT64。
  - n（int64_t，入参）：表示矩阵列数，数据类型支持INT64。
  - out（aclTensor*，计算输出）：表示计算结果，公式中的out，Device侧的aclTensor，Atlas A2 训练系列产品/Atlas 800I A2推理产品数据类型支持COMPLEX64，shape维度不高于3维，数据格式支持ND。不支持非连续的Tensor，不支持空Tensor。
  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。  
- **返回值**：
  aclnnStatus：返回状态码。
  
  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）: 传入的matx、maty或out是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）: matx、maty或out的数据类型不支持。
  ```

## aclnnComplexMatDot
- **参数说明**：
  - workspace（void \*, 入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t, 入参）：在Device侧申请的workspace大小，由第一段接口aclnnComplexMatDotGetWorkspaceSize获取。
  - executor（aclOpExecutor \*, 入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream, 入参）：指定执行任务的AscendCL Stream流。

- **返回值**：
  aclnnStatus：返回状态码。

## 约束与限制
无
