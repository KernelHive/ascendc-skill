# aclnnIndexAdd

## 支持的产品型号
- Atlas 推理系列产品
- Atlas 训练系列产品
- Atlas A2 训练系列产品/Atlas 800I A2 推理产品/Atlas A3 训练系列产品

## 接口原型

每个算子分为两段式接口，必须先调用“aclnnIndexAddGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnIndexAdd”接口执行计算。

- `aclnnStatus aclnnIndexAddGetWorkspaceSize(const aclTensor* self, const int64_t dim, const aclTensor* index, const aclTensor* source, const aclScalar* alpha, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnIndexAdd(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## 功能描述

算子功能：在指定维度上，根据给定的索引，将源张量中的值加到输入张量中对应位置的值上。


## aclnnIndexAddGetWorkspaceSize

- **参数说明：**

  - self（aclTensor\*，计算输入）：输入张量，Device侧的aclTensor，数据类型需要和source一致。支持非连续的Tensor，数据格式支持ND，数据维度支持0-8维。
    - Atlas 推理系列产品、Atlas 训练系列产品：数据类型支持FLOAT、FLOAT16、INT32、INT16、INT8、UINT8、DOUBLE、INT64、BOOL。
    - Atlas A2 训练系列产品/Atlas A3 训练系列产品/Atlas 800I A2 推理产品：数据类型支持FLOAT、FLOAT16、INT32、INT16、INT8、UINT8、DOUBLE、INT64、BOOL和BFLOAT16。

  - dim（int64_t，计算输入）: 指定的维度。数据类型支持INT64，取值范围为[-self.dim(), self.dim()-1]。

  - index（aclTensor\*，计算输入）：索引，Device侧的aclTensor，index的shape大小和source在dim维度上的shape值需要相等，数据类型支持INT64、INT32。数据格式支持ND，数据维度支持1维。

  - source（aclTensor\*，计算输入）：源张量，Device侧的aclTensor，数据类型需要和self一致，source的shape除dim维度之外，其他维度的值需要与self的shape相等。支持非连续的Tensor，数据格式支持ND，数据维度支持0-8维。
    - Atlas 推理系列产品、Atlas 训练系列产品：数据类型支持FLOAT、FLOAT16、INT32、INT16、INT8、UINT8、DOUBLE、INT64、BOOL。
    - Atlas A2 训练系列产品/Atlas A3 训练系列产品/Atlas 800I A2 推理产品：数据类型支持FLOAT、FLOAT16、INT32、INT16、INT8、UINT8、DOUBLE、INT64、BOOL和BFLOAT16。

  - alpha（aclScalar\*，计算输入）：host侧的aclScalar，数据类型可转换成self与source推导后的数据类型。
    - Atlas 推理系列产品、Atlas 训练系列产品：数据类型支持FLOAT、FLOAT16、INT32、INT16、INT8、UINT8、DOUBLE、INT64、BOOL。
    - Atlas A2 训练系列产品/Atlas A3 训练系列产品/Atlas 800I A2 推理产品：数据类型支持FLOAT、FLOAT16、INT32、INT16、INT8、UINT8、DOUBLE、INT64、BOOL和BFLOAT16。

  - out（aclTensor\*，计算输出）：输出Tensor，Device侧的aclTensor。shape需要与self的shape一致，数据类型需要和self一致，支持非连续的Tensor，数据格式支持ND，数据维度支持0-8维。
    - Atlas 推理系列产品、Atlas 训练系列产品：数据类型支持FLOAT、FLOAT16、INT32、INT16、INT8、UINT8、DOUBLE、INT64、BOOL。
    - Atlas A2 训练系列产品/Atlas A3 训练系列产品/Atlas 800I A2 推理产品：数据类型支持FLOAT、FLOAT16、INT32、INT16、INT8、UINT8、DOUBLE、INT64、BOOL和BFLOAT16。

  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
  
  - executor（aclOpExecutor\**，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：传入的self、index、source、alpha、out是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. self、index、source、out的数据类型不在支持的范围之内。
                                        2. self、source和out的数据类型不一致。
                                        3. dim的值大于self的shape大小。
                                        4. 计算出的数据类型无法转换为指定输出out的数据类型。
                                        5. self和source除了在维度dim上，其余维度上存在shape值不相等。
                                        6. index不为1维。
                                        7. index的shape大小和source在维度dim上的shape值不相等。
                                        8. out的shape和self的shape不相等。
  ```

## aclnnIndexAdd

- **参数说明：**

  - workspace（void\*，入参）：在Device侧申请的workspace内存地址。

  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnIndexAddGetWorkspaceSize获取。

  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。

  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。

- **返回值：**

  aclnnStatus：返回状态码。

## 约束与限制

index输入取值范围在[0，self.shape[dim])范围内，即索引值输入范围为self在dim维度上的shape大小。

## 调用示例

详见[InplaceIndexAddWithSorted自定义算子样例说明算子调用章节](../README.md#算子调用)