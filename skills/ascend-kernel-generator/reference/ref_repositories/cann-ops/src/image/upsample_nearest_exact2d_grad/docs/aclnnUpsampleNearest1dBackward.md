# aclnnUpsampleNearest1dBackward

## 支持的产品型号
- Atlas 推理系列产品。
- Atlas 训练系列产品。
- Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件。
- Atlas A3 训练系列产品/Atlas A3 推理系列产品。

## 接口原型
每个算子分为两段式接口，必须先调用“aclnnUpsampleNearest1dBackwardGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnUpsampleNearest1dBackward”接口执行计算。

- `aclnnStatus aclnnUpsampleNearest1dBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclIntArray *outputSize, const aclIntArray *inputSize, double scales, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnUpsampleNearest1dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`


## 功能描述

算子功能：aclnnUpsampleNearest1d的反向传播。

## aclnnUpsampleNearest1dBackwardGetWorkspaceSize

- **参数说明：**

  - gradOut（aclTensor\*，计算输入）：Device侧的aclTensor。表示反向计算的的梯度Tensor。支持非连续的Tensor。L轴不支持空Tensor。数据格式支持NCL，ND（当数据格式为ND时，默认按照NCL格式处理）。输入必须是3维。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持DOUBLE、FLOAT、BFLOAT16和FLOAT16。
    - Atlas 推理系列产品、Atlas 训练系列产品：数据类型支持FLOAT16。
  - outputSize（aclIntArray\*，计算输入）：Host侧的aclIntArray，size大小为1。表示输入gradOut在L维度上的空间大小。
  - inputSize（aclIntArray\*，计算输入）：Host侧的aclIntArray，size大小为3。表示输出out分别在N、C、L维度上的空间大小。
  - scales（double，计算输入）：Host侧的double变量，表示输出out的L维度乘数。
  - out（aclTensor\*，计算输出）：Device侧的aclTensor。表示反向计算的输出张量。支持非连续的Tensor。L轴不支持空Tensor。数据格式支持NCL，ND。输出必须是3维。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持DOUBLE、FLOAT、BFLOAT16和FLOAT16，且数据类型与gradOut的数据类型一致。
    - Atlas 推理系列产品、Atlas 训练系列产品：数据类型支持FLOAT16，且数据类型与gradOut的数据类型一致。
  - workspaceSize（uint64_t\*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）: 1. 传入的gradOut、outputSize、inputSize或out是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）: 1. gradOut的数据类型和数据格式不在支持的范围之内。
                                        2. gradOut和out的数据类型不一致。
                                        3. gradOut的维度不为3维。
                                        4. outputSize的size大小不等于1。
                                        5. outputSize的某个元素值小于1。
                                        6. inputSize的size大小不等于3。
                                        7. inputSize的某个元素值小于1。
                                        8. gradOut与inputSize在N、C维度上的size大小不同。
                                        9. gradOut在L维度上的size大小与outputSize[0]不同。
                                        10. gradOut和out的N/C轴的维度大小不相等。
  ```


## aclnnUpsampleNearest1dBackward

- **参数说明**

  - workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnUpsampleNearest1dBackwardGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。

- **返回值：**

  aclnnStatus：返回状态码。

## 约束与限制

参数outputSize与参数scales，在使用时二选一，即：
- 当入参scales的值为0时，使用入参outputSize的参数值。
- 当入参scales的值不为0时，使用入参scales参数值，且$outputSize=[floor(inputSizeL*scales)]$。

