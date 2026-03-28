# aclnnAscendQuant

## 支持的产品型号

- Atlas 推理系列产品。
- Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件。

## 接口原型
每个算子分为两段式接口，必须先调用“aclnnAscendQuantGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnAscendQuant”接口执行计算。

- `aclnnStatus aclnnAscendQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *offset, bool sqrtMode, const char *roundMode, int32_t dstType, const aclTensor *y, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnAscendQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- 算子功能：对输入x进行量化操作，scale和offset的size需要是x的最后一维或1。
- 计算公式：
  
  $$
  y = round((x * scale) + offset)
  $$
  sqrt\_mode为true时，计算公式为:
  $$
  y = round((x * scale * scale) + offset)
  $$

## aclnnAscendQuantGetWorkspaceSize

- **参数说明：**

  - x（aclTensor*，计算输入）：公式中的`x`，Device侧的aclTensor，需要做量化的输入。支持非连续的Tensor，不支持空tensor。数据格式支持ND。如果dstType为3，Shape的最后一维需要能被8整除；如果dstType为29，Shape的最后一维需要能被2整除。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
    - Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
  - scale（aclTensor*，计算输入）：公式中的`scale`，Device侧的aclTensor，量化中的scale值。scale支持1维张量或多维张量（当shape为1维张量时，scale的第0维需要等于x的最后一维或等于1；当shape为多维张量时，scale的维度需要和x保持一致，最后一个维度的值需要和x保持一致，其他维度为1）。支持非连续的Tensor，不支持空tensor。数据格式支持ND。如果x的dtype不是FLOAT32，需要和x的dtype一致。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
    - Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
  - offset（aclTensor*，计算输入）：公式中的`offset`，可选参数，Device侧的aclTensor，反向量化中的offset值。offset支持1维张量或多维张量（当shape为1维张量时，scale的第0维需要等于x的最后一维或等于1；当shape为多维张量时，scale的维度需要和x保持一致，最后一个维度的值需要和x保持一致，其他维度为1）。数据类型和shape需要与scale保持一致。支持非连续的Tensor，不支持空tensor。数据格式支持ND。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
    - Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
  - sqrtMode（bool，计算输入）：host侧的bool，指定scale参与计算的逻辑，当取值为true时，公式为y = round((x * scale * scale) + offset)。数据类型支持BOOL。
  - roundMode（char\*，计算输入）：host侧的string，指定cast到int8输出的转换方式，支持取值round/ceil/trunc/floor。
  - dstType（int32_t，计算输入）：host侧的int32_t，指定输出的数据类型，该属性数据类型支持INT。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件：支持取值2、3、29，分别表示INT8、INT32、INT4。
    - Atlas 推理系列产品：支持取值2，表示INT8。
  - y（aclTensor\*，计算输出）：公式中的`y`，Device侧的aclTensor。类型为INT32时，Shape的最后一维是x最后一维的1/8，其余维度和x一致; 其他类型时，Shape与x一致。支持非连续的Tensor，不支持空tensor。数据格式支持ND。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件：数据类型支持INT8、INT32、INT4。
    - Atlas 推理系列产品：数据类型支持INT8。
  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的x、scale、y是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. x、scale、offset、y的数据类型或数据格式不在支持的范围之内。
                                        2. x、scale、offset、y的shape不满足限制条件。
                                        3. roundMode不在有效取值范围。
                                        4. dstType不在有效取值范围。
                                        5. y的数据类型为int4时，x的shape尾轴大小不是偶数。
                                        6. y的数据类型为int32时，y的shape尾轴不是x的shape尾轴大小的8倍，或者x与y的shape的非尾轴的大小不一致。
  ```

## aclnnAscendQuant

- **参数说明：**

  - workspace（void*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnAscendQuantGetWorkspaceSize获取。
  - executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。

- **返回值：**

  aclnnStatus：返回状态码。

## 约束与限制

无。

