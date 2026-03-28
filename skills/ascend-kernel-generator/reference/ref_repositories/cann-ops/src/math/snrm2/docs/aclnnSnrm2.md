# aclnnSnrm2

## 支持的产品型号
- Atlas A2 训练系列产品/Atlas 800I A2 推理产品。

## 接口原型
每个算子分为两段式接口，必须先调用“aclnnSnrm2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSnrm2”接口执行计算。

- `aclnnStatus aclnnSnrm2GetWorkspaceSize(const aclTensor *x, int64_t n, int64_t incx, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnSnrm2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述
- 算子功能：计算实数向量的欧式范数。  
- 计算公式：  
  $$
  out = \sqrt{\sum_{i=1}^n |x_i|^2}
  $$
  其中，x为输入向量，out为结果向量。

## aclnnSnrm2GetWorkspaceSize
- **参数说明**：
  
  - x（aclTensor*，计算输入）：公式中的x，Device侧的aclTensor，Atlas A2 训练系列产品/Atlas 800I A2推理产品数据类型支持FLOAT32，shape维度不高于3维，数据格式支持ND。不支持非连续的Tensor，不支持空Tensor。
  - n（int64_t，入参）：向量x中得元素个数，数据类型支持INT64。
  - incx（int64_t，入参）：表示向量x相邻元素间的内存地址偏移量，支持数值为1，数据类型支持INT64。
  - out（aclTensor*，计算输出）：表示计算结果，公式中的out，Device侧的aclTensor，Atlas A2 训练系列产品/Atlas 800I A2推理产品数据类型支持FLOAT32，只包含1个元素，shape维度不高于3维，数据格式支持ND。不支持非连续的Tensor，不支持空Tensor。
  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。  
- **返回值**：
  aclnnStatus：返回状态码。
  
  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）: 传入的x或out是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）: x或out的数据类型不支持。
  ```

## aclnnSnrm2
- **参数说明**：
  - workspace(void \*, 入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnSnrm2GetWorkspaceSize获取。
  - executor(aclOpExecutor \*, 入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream, 入参)：指定执行任务的AscendCL Stream流。

- **返回值**：
  aclnnStatus：返回状态码。

## 约束与限制
无

