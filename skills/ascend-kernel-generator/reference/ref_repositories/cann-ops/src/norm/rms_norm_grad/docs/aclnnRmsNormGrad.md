# aclnnRmsNormGrad

## 支持的产品型号

- Atlas 推理系列产品。
- Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件。
- Atlas A3 训练系列产品/Atlas A3 推理系列产品。

## 接口原型
每个算子分为两段式接口，必须先调用`aclnnRmsNormGradGetWorkspaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnRmsNormGrad`接口执行计算。

- `aclnnStatus aclnnRmsNormGradGetWorkspaceSize(const aclTensor *dy, const aclTensor *x, const aclTensor *rstd, const aclTensor *gamma, const aclTensor *dxOut, const aclTensor *dgammaOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnRmsNormGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

* 算子功能：aclnnRmsNorm的反向计算。
* 算子公式：

  - 正向公式：

    $$
    \operatorname{RmsNorm}(x_i)=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} g_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}
    $$

  - 反向推导：

    $$
    dx_i= (dy_i * g_i - \frac{x_i}{\operatorname{Rms}(\mathbf{x})} * \operatorname{Mean}(\mathbf{y})) * \frac{1} {\operatorname{Rms}(\mathbf{x})},  \quad \text { where } \operatorname{Mean}(\mathbf{y}) = \frac{1}{n}\sum_{i=1}^n (dy * g_i * \frac{x_i}{\operatorname{Rms}(\mathbf{x})})
    $$

    $$
    dg_i = \frac{x_i}{\operatorname{Rms}(\mathbf{x})} dy_i
    $$

## aclnnRmsNormGradGetWorkspaceSize
- **参数说明：**
  - dy（aclTensor\*，计算输入）：公式中的输入`dy`，Device侧的aclTensor，表示反向传回的梯度。数据格式支持ND，shape支持1-8维度。
    - Atlas 推理系列产品：数据类型支持FLOAT32，FLOAT16。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32，FLOAT16，BFLOAT16。
  - x（aclTensor\*，计算输入）：公式中的输入`x`，Host侧的aclTensor，正向算子的输入，表示被标准化的数据。数据格式支持ND，shape支持1-8维度，且与入参`dy`的shape一致。
    - Atlas 推理系列产品：数据类型支持FLOAT32，FLOAT16。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32，FLOAT16，BFLOAT16。
  - rstd（aclTensor\*，计算输入）：公式中的输入`Rms(x)`，Host侧的aclTensor，正向算子的中间计算结果。数据类型支持FLOAT32。数据格式支持ND，shape支持1-8维度，shape需要满足rstd_shape = x_shape\[0:n\]，n < x_shape.dims()，n与gamma一致。
  - gamma（aclTensor\*，计算输入）：公式中的输入`g`，Host侧的aclTensor，正向算子的输入。数据格式支持ND，shape支持1-8维度，shape需要满足gamma_shape = x_shape\[n:\], n < x_shape.dims()。
    - Atlas 推理系列产品：数据类型支持FLOAT32，FLOAT16。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32，FLOAT16，BFLOAT16。
  - dxOut（aclTensor\*，计算输出）：公式中的输出`dx`，Host侧的aclTensor，表示输入`x`的梯度。数据格式支持ND，shape支持1-8维度，shape与入参`dy`的shape保持一致。
    - Atlas 推理系列产品：数据类型支持FLOAT32，FLOAT16。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32，FLOAT16，BFLOAT16。
  - dgammaOut（aclTensor\*，计算输出）：公式中的输出`dg`，Device侧的aclTensor，表示`gamma`的梯度。数据类型支持FLOAT32。数据格式支持ND，shape支持1-8维度，shape与入参`gamma`的shape保持一致。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）: 1. 如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  返回161002（ACLNN_ERR_PARAM_INVALID）: 1. 输入或输出的数据类型不在支持范围之内。
  返回561002（ACLNN_ERR_INNER_TILING_ERROR）: 1. 参数不满足参数说明中的要求。
  ```

## aclnnRmsNormGrad
- **参数说明：**

  - workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnRmsNormGradGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。

- **返回值：**
  aclnnStatus：返回状态码。

## 约束与限制
- **功能维度**
  - 数据类型支持
    - 入参 `dy`、`x`、`gamma`支持：
      - Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
      - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
    - 入参 `rstd`支持：FLOAT32。
  - Atlas 推理系列产品：`x`、`dy`、`gamma`输入的尾轴长度必须大于等于 32 Bytes。
  - 数据格式支持：ND。
- **未支持类型说明**
  - DOUBLE：指令不支持DOUBLE。
  - 是否支持空Tensor：不支持空进空出。
  - 是否非连续的Tensor：不支持输入非连续。
- **边界值场景说明**
  - Atlas 推理系列产品：输入不支持包含inf和nan
  **各产品支持数据类型说明**
  - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：
    | `dy` 数据类型 | `x` 数据类型 | `rstd` 数据类型 | `gamma` 数据类型 | `dx` 数据类型 | `dgamma` 数据类型 |
    | -------- | -------- | -------- | -------- | -------- | -------- |
    | float16  | float16  | float32  | float32  | float16  | float32  |
    | bfloat16 | bfloat16 | float32  | float32  | bfloat16 | float32  |
    | float16  | float16  | float32  | float16  | float16  | float32  |
    | float32  | float32  | float32  | float32  | float32  | float32  |
    | bfloat16 | bfloat16 | float32  | bfloat16 | bfloat16 | float32  |
  - Atlas 推理系列产品：
    | `dy` 数据类型 | `x` 数据类型 | `rstd` 数据类型 | `gamma` 数据类型 | `dx` 数据类型 | `dgamma` 数据类型 |
    | -------- | -------- | -------- | -------- | -------- | -------- |
    | float16  | float16  | float32  | float16  | float16  | float32  |
    | float32  | float32  | float32  | float32  | float32  | float32  |
