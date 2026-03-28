# aclnnUpsampleNearestExact2d

## 支持的产品型号

- Atlas 推理系列产品。
- Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件。
- Atlas A3 训练系列产品/Atlas A3 推理系列产品。

## 接口原型

每个算子分为两段式接口，必须先调用“aclnnUpsampleNearestExact2dGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnUpsampleNearestExact2d”接口执行计算。
- `aclnnStatus aclnnUpsampleNearestExact2dGetWorkspaceSize(const aclTensor *self, const aclIntArray *outputSize, double scalesH, double scalesW, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnUpsampleNearestExact2d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- 算子功能：对由四个输入通道组成的输入信号应用最近邻精确插值算法进行上采样。如果输入shape为（N，C，H，W），则输出shape为（N，C，outputSize[0]，outputSize[1]）。
- 计算公式：

  $$
  h_{src} = min(floor((h_{dst} + 0.5) * scalesH),  H - 1)
  $$

  $$
  w_{src} = min(floor((w_{dst} + 0.5) * scalesW),  W - 1)
  $$

  $$
  out(N, C, h_{dst}, w_{dst}) = self(N, C, h_{src}, w_{src})
  $$

## aclnnUpsampleNearestExact2dGetWorkspaceSize

- **参数说明**：

  - self（aclTensor*，计算输入）：公式中的输入`self`，Device侧的aclTensor。表示进行上采样的输入张量。支持非连续的Tensor，不支持空Tensor。数据格式支持NCHW、ND（当数据格式为ND时，默认按照NCHW格式处理）。输入维度必须是4。
    - Atlas 推理系列产品：数据类型支持FLOAT、FLOAT16。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT、FLOAT16、BFLOAT16。
  - outputSize（aclIntArray*,计算输入）：Host侧的aclIntArray，指定输出tensor大小，数据类型支持INT64。
  - scalesH（double，计算输入）：公式中的输入`scalesH`，Host侧的DOUBLE型参数，指定H方向空间大小的缩放乘数。
  - scalesW（double，计算输入）：公式中的输入`scalesW`，Host侧的DOUBLE型参数，指定W方向空间大小的缩放乘数。
  - out（aclTensor\*，计算输出）：公式中的输出`out`，Device侧的aclTensor。表示采样后的输出张量。支持非连续的Tensor，不支持空Tensor。数据格式支持NCHW、ND。数据类型与入参`self`的数据类型保持一致。
    - Atlas 推理系列产品：数据类型支持FLOAT、FLOAT16。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT、FLOAT16、BFLOAT16。
  - workspaceSize（uint64_t\*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
  
- **返回值**：

  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR): 1. 传入的self或out是空指针。
  返回161002 (ACLNN_ERR_PARAM_INVALID): 1. self的数据类型不在支持的范围内或self与out数据类型不同。
                                        2. self的数据格式不在支持范围内。
                                        3. self的shape不是4维。
                                        4. self和out的N/C轴的维度大小不相等。
                                        5. self和out的数据格式不在支持的范围之内。
  ```

## aclnnUpsampleNearestExact2d

- **参数说明**：

  - workspace（void\*, 入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t, 入参）：在Device侧申请的workspace大小，由第一段接口aclnnUpsampleNearestExact2dGetWorkspaceSize获取。
  - executor（aclOpExecutor\*, 入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream, 入参）：指定执行任务的AscendCL Stream流。

- **返回值**：

  **aclnnStatus**：返回状态码。

## 约束与限制

参数outputSize与参数scalesH、scalesW，在使用时二选一，即：
- 当入参scalesH和入参scalesW的值都为0时，使用入参outputSize的参数值。
- 当入参scalesH和入参scalesW的值不都为0时，使用入参scalesH和入参scalesW的参数值，且$outputSize=[floor(selfH*scalesH)，floor(selfW*scalesW)]$。
