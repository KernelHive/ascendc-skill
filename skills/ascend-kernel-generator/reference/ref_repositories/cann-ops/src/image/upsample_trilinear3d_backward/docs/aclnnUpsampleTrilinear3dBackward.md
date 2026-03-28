# aclnnUpsampleTrilinear3dBackward

## 支持的产品型号

- Atlas 训练系列产品。
- Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件。
- Atlas A3 训练系列产品/Atlas A3 推理系列产品。

## 接口原型

每个算子分为两段式接口，必须先调用“aclnnUpsampleTrilinear3dBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnUpsampleTrilinear3dBackward”接口执行计算。
- `aclnnStatus aclnnUpsampleTrilinear3dBackwardGetWorkspaceSize(const aclTensor* gradOut, const aclIntArray* outputSize, const aclIntArray* inputSize, bool alignCorners, double scalesD, double scalesH, double scalesW, aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnUpsampleTrilinear3dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

算子功能：aclnnUpsampleTrilinear3d的反向计算。

## aclnnUpsampleTrilinear3dBackwardGetWorkspaceSize

- **参数说明**：

  - gradOut（aclTensor*，计算输入）：Device侧的aclTensor。表示反向计算的的梯度Tensor。支持非连续的Tensor，不支持空Tensor。数据格式支持NCDHW、NDHWC、ND（当数据格式为ND时，默认按照NCDHW格式处理）。shape仅支持五维。
    - Atlas 训练系列产品：数据类型支持FLOAT、FLOAT16、DOUBLE。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT、FLOAT16、BFLOAT16、DOUBLE。
  - outputSize（aclIntArray*，计算输入）：Host侧的aclIntArray，数据类型支持INT64，size大小为3。表示输入`gradOut`在D、H和W维度上的空间大小。
  - inputSize（aclIntArray*，计算输入）：Host侧的aclIntArray，数据类型支持INT64，size大小为5。表示输出`gradInput`分别在N、C、D、H、和W维度上的空间大小。
  - alignCoreners（bool，计算输入）: Host侧的bool类型参数，是否对齐角像素点。如果为true，则输入和输出张量的角像素点会被对齐，否则不对齐。
  - scalesD（double，计算输入）：Host侧的double常量，表示输出`gradInput`的depth维度乘数。
  - scalesH（double，计算输入）：Host侧的double常量，表示输出`gradInput`的height维度乘数。
  - scalesW（double，计算输入）：Host侧的double常量，表示输出`gradInput`的width维度乘数。
  - gradInput（aclTensor*，计算输出）：Device侧的aclTensor。表示反向计算的输出张量。shape仅支持五维，shape在N、C、D、H、和W维度上的大小需与`inputSize`中给定的N、C、D、H、和W维度上的空间大小一致。支持非连续的Tensor，不支持空Tensor。数据格式支持NCDHW、NDHWC、ND。数据类型和数据格式与入参`gradOut`的数据类型和数据格式保持一致。
    - Atlas 训练系列产品：数据类型支持FLOAT、FLOAT16、DOUBLE。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT、FLOAT16、BFLOAT16、DOUBLE。
  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。

- **返回值**：

  aclnnStatus：返回状态码。

```
第一段接口完成入参校验，出现以下场景时报错：
返回161001（ACLNN_ERR_PARAM_NULLPTR）: 1. 传入的gradOut、outputSize、inputSize或gradInput是空指针。
返回161002（ACLNN_ERR_PARAM_INVALID）: 1. gradOut的数据类型和数据格式不在支持的范围内。
                                      2. gradOut和gradInput的数据类型不一致。
                                      3. gradOut的维度不为5维。
                                      4. outputSize的size大小不等于3。
                                      5. outputSize的某个元素值不大于0。
                                      6. inputSize的size大小不等于5。
                                      7. inputSize的某个元素值不大于0。
                                      8. gradOut与inputSize在N、C维度上的size大小不同。
                                      9. gradOut在D、H、W维度上的size大小与outputSize[0]、outputSize[1]和outputSize[2]未完全相同。
                                      10. gradInput的shape与inputSize[0]、inputSize[1]、inputSize[2]、inputSize[3]和inputSize[4]未完全相同。
```

## aclnnUpsampleTrilinear3dBackward

- **参数说明**：

  - workspace（void*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnUpsampleTrilinear3dBackwardGetWorkspaceSize获取。
  - executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。

- **返回值**：

  **aclnnStatus**：返回状态码。

## 约束与限制

无。
