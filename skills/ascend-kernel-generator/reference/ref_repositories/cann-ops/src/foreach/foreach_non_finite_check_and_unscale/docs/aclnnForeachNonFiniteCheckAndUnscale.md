# aclnnForeachNonFiniteCheckAndUnscale

## 支持的产品型号

- Atlas A2 训练系列产品。

## 接口原型

每个算子分为两段式接口，必须先调用“aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnForeachNonFiniteCheckAndUnscale”接口执行计算。

- `aclnnStatus aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize(const aclTensorList *scaledGrads, const aclTensor *foundInf, const aclTensor *inScale, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnForeachNonFiniteCheckAndUnscale(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- 算子功能：遍历scaledGrads中的所有Tensor，检查是否存在inf或NaN，如果存在则将foundInf设置为1.0，否则foundInf保持不变，并对scaledGrads中的所有Tensor进行反缩放。

- 计算公式：
  $$
  foundInf = \begin{cases}1.0, & 当 inf \in  scaledGrads 或 Nan \in scaledGrads,\\
    foundInf, &其他.
  \end{cases}
  $$

  $$
   scaledGrads_i = {scaledGrads}_{i}*{invScale}.
  $$

## aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize

- **参数说明**：

  - scaledGrads（aclTensorList*，计算输入/输出）：公式中的`scaledGrads`，Device侧的aclTensorList，表示进行反缩放计算的输入和输出张量列表。数据类型支持FLOAT、FLOAT16、BFLOAT16。shape维度不高于8维，数据格式支持ND。支持非连续的Tensor，支持的最大长度为256个。
  - foundInf（aclTensor*，计算输入/输出）：公式中的`foundInf`，Device侧的aclTensor，仅包含一个元素。表示用来标记输入scaledGrads中是否存在inf或-inf的标量。数据类型支持FLOAT，数据格式支持ND。不支持非连续的Tensor。
  - inScale（aclTensor*，计算输入）：公式中的`inScale`，Device侧的aclTensor，仅包含一个元素。表示进行反缩放计算的标量。数据类型支持FLOAT，数据格式支持ND。不支持非连续的Tensor。
  - workspaceSize（uint64_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\**，出参）：返回op执行器，包含了算子计算流程。

- **返回值**：

  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的scaledGrads、foundInf、inScale是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. scaledGrads、foundInf、inScalet的数据类型不在支持的范围之内。
  返回561002（ACLNN_ERR_INNER_TILING_ERROR）：1. scaledGrads长度超过限制。
                                             2. scaledGrads中的Tensor数据类型不相同。                                  
  ```

## aclnnForeachNonFiniteCheckAndUnscale

- **参数说明**：

  - workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。

- **返回值**：

  aclnnStatus：返回状态码。

## 约束与限制

无。
