# aclnnUpsampleLinear1d

## 支持的产品型号

- Atlas 训练系列产品。
- Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件。
- Atlas A3 训练系列产品/Atlas A3 推理系列产品。

## 接口原型
每个算子分为两段式接口，必须先调用“aclnnUpsampleLinear1dGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnUpsampleLinear1d”接口执行计算。

- `aclnnStatus aclnnUpsampleLinear1dGetWorkspaceSize(const aclTensor *self, const aclIntArray *outputSize, const bool alignCorners, const double scales, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnUpsampleLinear1d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

算子功能：对由多个输入通道组成的输入信号应用线性插值算法进行上采样。如果输入shape为 (N，C，L) ，则输出shape为 (N，C，outputSize) 。

## aclnnUpsampleLinear1dGetWorkspaceSize

- **参数说明**

  - self（aclTensor\*，计算输入）：Device侧的aclTensor，表示进行上采样的输入张量。支持非连续的Tensor，不支持空Tensor。数据格式支持NCL。输入维度必须是3.
    - Atlas 训练系列产品：数据类型支持FLOAT、FLOAT16。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT、FLOAT16、BFLOAT16。
  - outputSize（aclIntArray\*，计算输入）：Host侧的aclIntArray，size大小为1。表示输出out在L维度上的空间大小。
  - alignCorners（bool，计算输入）：Host侧的bool类型参数。如果设置为True，则输入和输出张量按其角像素的中心点对齐，保留角像素处的值；如果设置为False，则输入和输出张量通过其角像素的角点对齐，并且插值使用边缘值填充用于外界边值，使此操作在保持不变时独立于输入大小scales。
  - scales（double, 计算输入）：Host侧的double常量，表示输出out的L维度乘数。
  - out（aclTensor\*，计算输出）：Device侧的aclTensor，表示采样后的输出张量。支持非连续的Tensor，不支持空Tensor。数据格式支持NCL。输出维度必须是3。数据类型与入参`self`的数据类型一致。
    - Atlas 训练系列产品：数据类型支持FLOAT、FLOAT16。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT、FLOAT16、BFLOAT16。
  - workspaceSize（uint64_t\*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）: 1. 传入的self、outputSize或out是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）: 1. self的数据类型和数据格式不在支持的范围之内。
                                        2. self和out的数据类型不一致。
                                        3. self和out的维度不为3维。
                                        4. outputSize的size大小不等于1。
                                        5. outputSize的某个元素值小于1。
                                        6. out在L维度上的size大小与outputSize[0]未完全相同。
  ```


## aclnnUpsampleLinear1d

- **参数说明**

  - workspace（void\*, 入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t, 入参）：在Device侧申请的workspace大小，由第一段接口aclnnUpsampleLinear1dGetWorkspaceSize获取。
  - executor（aclOpExecutor\*, 入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream, 入参）：指定执行任务的AscendCL Stream流。

- **返回值：**

  aclnnStatus：返回状态码。

## 约束与限制

参数outputSize与参数scales，在使用时二选一，即：
- 当入参scales的值为0时，使用入参outputSize的参数值。
- 当入参scales的值不为0时，使用入参scales参数值，且$outputSize=[floor(selfL*scales)]$。


