# aclnnScatterValue&aclnnInplaceScatterValue

## 支持的产品型号

- Atlas 训练系列产品/Atlas A2 训练系列产品/Atlas A3 训练系列产品/Atlas 800I A2 推理产品

## 接口原型

- aclnnScatterValue和aclnnInplaceScatterValue实现相同的功能，使用区别如下，请根据自身实际场景选择合适的算子。

  - aclnnScatterValue：需新建一个输出张量对象存储计算结果。
  - aclnnInplaceScatterValue：无需新建输出张量对象，直接在输入张量的内存中存储计算结果。

- 每个算子分为两段式接口，必须先调用“aclnnScatterValueGetWorkspaceSize”或者“aclnnInplaceScatterValueGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnScatterValue”或者“aclnnInplaceScatterValue”接口执行计算。

  - `aclnnStatus aclnnScatterValueGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index, const aclScalar *value, int64_t reduce, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnScatterValue(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`
  - `aclnnStatus aclnnInplaceScatterValueGetWorkspaceSize(aclTensor *selfRef, int64_t dim, const aclTensor *index, const aclScalar *value, int64_t reduce, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnInplaceScatterValue(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述
- **算子功能**：将scalar value中的值按指定的轴和方向和对应的位置关系逐个填入tensor self中。value会被broadcast成和tensor index的shape一致的tensor src进行Scatter的计算。

- **示例：**
  对于一个3D tensor， self会按照如下的规则进行更新：

  ```
  self[index[i][j][k]][j][k] = value # 如果 dim == 0
  self[i][index[i][j][k]][k] = value # 如果 dim == 1
  self[i][j][index[i][j][k]] = value # 如果 dim == 2
  ```

  在计算时需要满足以下要求：
  - self、index维度数量必须相同。
  - 对于每一个维度d，如果d != dim, 有index.size(d) <= self.size(d)的限制。
  - dim的值的大小必须在 [-self的维度数量, self的维度数量-1] 之间。
  - self的维度数应该小于等于8。
  - index中对应维度dim的值大小必须在[0, self.size(dim)-1]之间。


## aclnnScatterValueGetWorkspaceSize
- **参数说明：**

  - self(aclTensor*, 计算输入)：公式中的`self`，Device侧的aclTensor。scatter的目标张量。self的维度数量需要与index相同,shape支持0-8维。支持非连续的Tensor支持ND。
    - Atlas 训练系列产品：数据类型支持UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128。
    - Atlas A2 训练系列产品/Atlas A3 训练系列产品/Atlas 800I A2 推理产品：数据类型支持UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16。

  - dim(int64_t, 计算输入)：用来scatter的维度，数据类型为INT64。范围为[-self的维度数量, self的维度数量-1]。

  - index(aclTensor*, 计算输入)：公式中的`index`，Device侧的aclTensor.索引张量，数据类型支持INT32、INT64。index的维度数量需要与self相同,shape支持0-8维。支持非连续的Tensor支持ND。

  - value(aclScalar*, 计算输入)：host侧的aclScalar。当value为COMPLEX时，self也必须为COMPLEX tensor，无其他数据类型限制。

  - reduce(int64_t, 计算输入)：选择应用的reduction操作。可选的操作选项以及对应的int值为 (add, 1), (mul, 2)，(none, 0)。具体操作含义如下：
    0：表示替换操作，将value按照index替换到out中的对应位置。
    1：表示累加操作，将value按照index累加到out中的对应位置。
    2：表示累乘操作，将value按照index累乘到out的对应位置。

  - out(aclTensor*, 计算输出)： scatter的输出结果将存储在此张量中。数据格式、数据类型、shape需要与self一致。
    - Atlas 训练系列产品：数据类型支持UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128。
    - Atlas A2 训练系列产品/Atlas A3 训练系列产品/Atlas 800I A2 推理产品：数据类型支持UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16。

  - workspaceSize(uint64_t *，出参)：返回需要在Device侧申请的workspace大小。

  - executor(aclOpExecutor ，出参)：返回op执行器，包含了算子计算流程。


- **返回值：**

  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR): 1. 传入的self、index、value或out是空指针。
  返回161002 (ACLNN_ERR_PARAM_INVALID): 1. self、index、value或out的数据类型不在支持范围内。
                                        2. self、out的数据类型不一样。
                                        3. self、index的维度数不一致。
                                        4. self和out的shape不一致。
                                        5. self、index的shape不符合以下限制：
                                          对于每一个维度d，如果d != dim, 有index.size(d) <= self.size(d)的限制。
                                        6. dim的值不在[-self的维度数量， self的维度数量-1]之间。
                                        7. self的维度数超过8。
                                        8. value为COMPLEX时，self的数据类型不为COMPLEX。
  ```

## aclnnScatterValue

- **参数说明：**

  - workspace(void*，入参)：在Device侧申请的workspace内存地址。

  - workspaceSize(uint64_t，入参)：在Device侧申请的workspace大小，由第一段接口aclnnScatterValueGetWorkspaceSize获取。

  - executor(aclOpExecutor *，入参)：op执行器，包含了算子计算流程。

  - stream(aclrtStream，入参)：指定执行任务的AscendCL Stream流。


- **返回值：**

  aclnnStatus：返回状态码。

## aclnnInplaceScatterValueGetWorkspaceSize

- **参数说明：**

  - selfRef(aclTensor*, 计算输入|计算输出)：公式中的`self`，Device侧的aclTensor。scatter的目标张量。selfRef的维度数量需要与index相同，shape支持0-8维。支持空tensor， 支持非连续的Tensor支持ND。
    - Atlas 训练系列产品：数据类型支持UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT、DOUBLE、COMPLEX64、COMPLEX128。
    - Atlas A2 训练系列产品/Atlas A3 训练系列产品/Atlas 800I A2 推理产品：数据类型支持UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16。
  - dim(int64_t, 计算输入)：用来scatter的维度，数据类型为INT64。范围为[-selfRef的维度数量, selfRef的维度数量-1]。
  - index(aclTensor*, 计算输入)：公式中的`index`，Device侧的aclTensor。索引张量，数据类型支持INT32、INT64。index的维度数量需要与selfRef相同,shape支持0-8维。对于每一个维度d，如果d != dim, 需保证index.size(d) <= selfRef.size(d)。支持空tensor，支持非连续的Tensor支持ND。
  - value(aclScalar*, 计算输入)：当value为COMPLEX时，selfRef也必须为COMPLEX tensor，无其他数据类型限制。
  - reduce(int64_t, 计算输入)：选择应用的reduction操作。可选的操作选项以及对应的int值为 (add, 1), (mul, 2)，(none, 0)。具体操作含义如下：
    0：表示替换操作，将value按照index替换到selfRef中的对应位置
    1：表示累加操作，将value按照index累加到selfRef中的对应位置
    2：表示累乘操作，将value按照index累乘到selfRef的对应位置
  - workspaceSize(uint64_t ，出参)：返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor **，出参)：返回op执行器，包含了算子计算流程。

- **返回值：**
  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR): 1. 传入的selfRef、index、value是空指针。
  返回161002 (ACLNN_ERR_PARAM_INVALID): 1. selfRef、index、value的数据类型不在支持范围内
                                        2. selfRef、out的数据类型不一样。
                                        3. selfRef、index的维度数不一致
                                        4. selfRef和out的shape不一致。
                                        5. selfRef、index的shape不符合以下限制：
                                         对于每一个维度d，如果d != dim, 有index.size(d) <= selfRef.size(d)的限制
                                        6. dim的值不在[-selfRef的维度数量， selfRef的维度数量-1]之间
                                        7. selfRef的维度数超过8
                                        8. value为COMPLEX时，selfRef的数据类型不为COMPLEX
  ```

## aclnnInplaceScatterValue

- **参数说明：**
  - workspace(void*，入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t，入参)：在Device侧申请的workspace大小，由第一段接口aclnnInplaceScatterValueGetWorkspaceSize获取。
  - executor(aclOpExecutor *，入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream，入参)：指定执行任务的AscendCL Stream流。

- **返回值：**
  aclnnStatus：返回状态码。

## 约束与限制

无。

## 调用示例

详见[ScatterAddWithSorted自定义算子样例说明算子调用章节](../README.md#算子调用)