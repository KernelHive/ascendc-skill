# aclnnRmsNorm

## 支持的产品型号

- Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件。
- Atlas 推理系列产品。
- Atlas A3 训练系列产品/Atlas A3 推理系列产品。

## 接口原型

每个算子分为两段式接口，必须先调用`aclnnRmsNormGetWorkspaceSize`接口获取入参并根据计算流程所需workspace大小，再调用`aclnnRmsNorm`接口执行计算。

- `aclnnStatus aclnnRmsNormGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, double epsilon, const aclTensor *yOut, const aclTensor *rstdOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnRmsNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- 算子功能：RmsNorm算子是大模型常用的归一化操作，相比LayerNorm算子，其去掉了减去均值的部分。
- 计算公式：

  $$
  \operatorname{RmsNorm}(x_i)=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} g_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}
  $$

## aclnnRmsNormGetWorkspaceSize

- **参数说明：**

  - x（aclTensor*，计算输入）：公式中的输入`x`，Device侧的aclTensor，shape支持1-8维度，数据格式支持ND。
    - Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  - gamma（aclTensor*，计算输入）：公式中的输入`g`，Host侧的aclTensor，shape支持1-8维度，数据格式支持ND。shape需要满足gamma_shape = x_shape\[n:\], n < x_shape.dims()。
    - Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  - epsilon（double，计算输入）：公式中的输入`eps`，Host侧的double型，用于防止除0错误，数据类型为double，缺省值为1e-6。
  - yOut（aclTensor*，计算输出）：公式中的输出`RmsNorm(x)`，Host侧的aclTensor，shape支持1-8维度，数据格式支持ND。shape与入参`x`的shape保持一致。
    - Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  - rstdOut（aclTensor*，计算输出）：公式中的输出`Rms(x)`，Device侧的aclTensor，数据类型支持FLOAT32、shape支持1-8维度，数据格式支持ND。shape与入参`x`的shape前几维保持一致，前几维指x的维度减去gamma的维度，表示不需要norm的维度。
  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）: 1. 如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  返回161002（ACLNN_ERR_PARAM_INVALID）: 1. 输入或输出的数据类型不在支持范围之内。
  返回561002（ACLNN_ERR_INNER_TILING_ERROR）: 1. 参数不满足参数说明中的要求。
  ```

## aclnnRmsNorm

- **参数说明：**

  - workspace（void*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnRmsNormGetWorkspaceSize获取。
  - executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

- **返回值：**

  aclnnStatus：返回状态码。

## 约束与限制
- **功能维度**
  - 数据类型支持
    - Atlas 推理系列产品：x、gamma支持FLOAT32、FLOAT16。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：x、gamma支持FLOAT32、FLOAT16、BFLOAT16。
    - rstd支持：FLOAT32。
  - 数据格式支持：ND。
  - Atlas 推理系列产品：x、gamma输入的尾轴长度必须大于等于 32 Bytes。
- **未支持类型说明**
  - DOUBLE：指令不支持DOUBLE。
  - 是否支持空tensor：不支持空进空出。
  - 是否支持非连续tensor：不支持输入非连续。
- **边界值场景说明**
  - Atlas 推理系列产品：
    输入不支持包含inf和nan。
  - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：
    当输入是inf时，输出为inf。
    当输入是nan时，输出为nan。
- **各平台支持数据类型说明**
  - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：
    | `x` 数据类型 | `gamma` 数据类型 | `y` 数据类型 | `rstd` 数据类型 |
    | -------- | -------- | -------- | -------- |
    | float16 | float32  | float16 | float32 |
    | bfloat16 | float32 | bfloat16 | float32 |
    | float16 | float16 | float16 | float32 |
    | bfloat16 | bfloat16 | bfloat16 | float32 |
    | float32 | float32  | float32 | float32  |
  - Atlas 推理系列产品：
    | `x` 数据类型 | `gamma` 数据类型 | `y` 数据类型 | `rstd` 数据类型 |
    | -------- | -------- | -------- | -------- |
    | float16 | float16 | float16 | float32 |
    | float32 | float32  | float32 | float32  |
