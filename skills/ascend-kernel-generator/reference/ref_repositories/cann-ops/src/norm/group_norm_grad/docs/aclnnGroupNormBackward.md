# aclnnGroupNormBackward

## 支持的产品型号

- Atlas 训练系列产品。
- Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件。
- Atlas A3 训练系列产品/Atlas A3 推理系列产品。

## 接口原型

每个算子分为两段式接口，必须先调用“aclnnGroupNormBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGroupNormBackward”接口执行计算。

  - `aclnnStatus aclnnGroupNormBackwardGetWorkspaceSize(const aclTensor* gradOut, const aclTensor* input, const aclTensor* mean, const aclTensor* rstd, const aclTensor* gamma, int64_t N, int64_t C, int64_t HxW, int64_t group, const aclBoolArray* outputMask, aclTensor* gradInput, aclTensor* gradGammaOut, aclTensor* gradBetaOut, uint64_t* workspaceSize, aclOpExecutor** executor)`
  - `aclnnStatus aclnnGroupNormBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

算子功能：aclnnGroupNorm的反向计算。

## aclnnGroupNormBackwardGetWorkspaceSize

- **参数说明：**

  - gradOut（aclTensor*，计算输入）：反向计算的梯度tensor，Device侧的aclTensor。shape支持0-8维，元素个数需要等于N\*C\*HxW，数据类型与input相同，数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
    - Atlas 训练系列产品：数据类型支持FLOAT16、FLOAT。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT16、FLOAT、BFLOAT16。
    
  - input（aclTensor*，计算输入）：正向计算的首个输入，Device侧的aclTensor。shape支持0-8维，元素个数需要等于N\*C\*HxW，数据类型与gradOut相同，数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
    - Atlas 训练系列产品：数据类型支持FLOAT16、FLOAT。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT16、FLOAT、BFLOAT16。

  - mean（aclTensor*，计算输入）：正向计算的第二个输出，表示input分组后每个组的均值，Device侧的aclTensor。shape支持0-8维，元素个数需要等于N\*group，数据类型与gradOut相同，数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
    - Atlas 训练系列产品：数据类型支持FLOAT16、FLOAT。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT16、FLOAT、BFLOAT16。

  - rstd（aclTensor*，计算输入）: 正向计算的第三个输出，表示input分组后每个组的标准差倒数，Device侧的aclTensor。shape支持0-8维，元素个数需要等于N\*group，数据类型与gradOut相同，数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
    - Atlas 训练系列产品：数据类型支持FLOAT16、FLOAT。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT16、FLOAT、BFLOAT16。

  - gamma（aclTensor*，计算输入）: 表示每个channel的缩放系数，Device侧的aclTensor。shape支持0-8维，元素个数需要等于C，数据类型与gradOut相同，数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
    - Atlas 训练系列产品：数据类型支持FLOAT16、FLOAT。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT16、FLOAT、BFLOAT16。

  - N（int64_t，计算输入）: INT64常量，表示输入gradOut在N维度上的空间大小。

  - C（int64_t，计算输入）: INT64常量，表示输入gradOut在C维度上的空间大小。

  - HxW（int64_t，计算输入）: INT64常量，表示输入gradOut在除N、C维度外的空间大小。

  - group（int64_t，计算输入）: INT64常量，表示将输入gradOut的C维度分为group组，group需大于0，且C必须可以被group整除。

  - outputMask（aclBoolArray*，计算输入）: 数据类型支持BOOL，size大小为3。分别表示是否输出gradInput，gradGammaOut，gradBetaOut，若为true则输出，否则输出对应位置返回空。

  - gradInput（aclTensor*，计算输出）: 输出张量，数据类型与gradOut相同，shape与input相同，Device侧的aclTensor。数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
    - Atlas 训练系列产品：数据类型支持FLOAT16、FLOAT。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT16、FLOAT、BFLOAT16。

  - gradGammaOut（aclTensor*，计算输出）: 输出张量，数据类型与gradOut相同，shape与gamma相同，Device侧的aclTensor。数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
    - Atlas 训练系列产品：数据类型支持FLOAT16、FLOAT。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT16、FLOAT、BFLOAT16。

  - gradBetaOut（aclTensor*，计算输出）: 输出张量，数据类型与gradOut相同，shape与gamma相同，Device侧的aclTensor。数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
    - Atlas 训练系列产品：数据类型支持FLOAT16、FLOAT。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT16、FLOAT、BFLOAT16。

  - workspaceSize（uint64_t*，出参）: 返回用户需要在npu device侧申请的workspace大小。

  - executor（aclOpExecutor**，出参）: 返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus: 返回状态码。
  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的 gradOut、input、mean、rstd是空指针时。
                                        2. 当outputMasked[0]为true，传入的 gradInput是空指针时。
                                        3. 当outputMasked[1]为true，传入的 gradGammaOut是空指针时。
                                        4. 当outputMasked[2]为true，传入的 gradBetaOut是空指针时。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. gradOut数据类型不在支持的范围之内。
                                        2. input、mean、out、rstd的数据类型与gradOut不同。
                                        3. 当outputMasked[0]为true，gradInput的shape与input的shape不相同。
                                        4. 当outputMasked[1]为true，gradGammaOut的shape与gamma的shape不相同。
                                        5. 当outputMasked[2]为true，gradBetaOut的shape与gamma的shape不相同。
                                        6. group不大于0。
                                        7. C不能被group整除。
                                        8. input的元素个数不等于 N * C * HxW。
                                        9. mean的元素个数不等于 N * group。
                                        10. rstd的元素个数不等于 N * group。
                                        11. gamma不为空指针且gamma的元素数量不为C。
    ```

## aclnnGroupNormBackward

- **参数说明：**

  - workspace（void*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnGroupNormBackwardGetWorkspaceSize获取。
  - executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：执行任务的AscendCL Stream流。

- **返回值：**

  aclnnStatus：返回状态码。

## 约束与限制

Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：在训练场景下C/group的结果不支持超过8000。
