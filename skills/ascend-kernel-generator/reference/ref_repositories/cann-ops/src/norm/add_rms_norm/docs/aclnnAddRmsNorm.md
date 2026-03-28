# aclnnAddRmsNorm

## 支持的产品型号

- Atlas 推理系列产品。
- Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件。
- Atlas A3 训练系列产品/Atlas A3 推理系列产品。

## 接口原型

每个算子分为两段式接口，必须先调用`aclnnAddRmsNormGetWorkspaceSize`接口获取入参并根据计算流程所需workspace大小，再调用`aclnnAddRmsNorm`接口执行计算。

- `aclnnStatus aclnnAddRmsNormGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *gamma, double epsilonOptional, const aclTensor *yOut, const aclTensor *rstdOut, const aclTensor *xOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnAddRmsNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- 算子功能：RmsNorm算子是大模型常用的归一化操作，相比LayerNorm算子，其去掉了减去均值的部分。
  AddRmsNorm算子将RmsNorm前的Add算子融合起来，减少搬入搬出操作。
- 计算公式：

  $$
  x_i=x_{i1}+x_{i2}
  $$

  $$
  \operatorname{RmsNorm}(x_i)=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} g_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}
  $$

## aclnnAddRmsNormGetWorkspaceSize

- **参数说明：**

  * x1（aclTensor*，计算输入）：公式中的输入`x1`，shape支持1-8维度，数据格式支持ND。不支持空Tensor。
    * Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
    * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  * x2（aclTensor*，计算输入）：公式中的输入`x2`，shape支持1-8维度，数据格式支持ND。shape需要与x1保持一致。不支持空Tensor。
    * Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
    * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  * gamma（aclTensor*，计算输入）：公式中的输入`g`，shape支持1-8维度，数据格式支持ND。shape需要与x1后几维保持一致，后几维表示需要norm的维度。不支持空Tensor。
    * Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
    * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  * epsilonOptional（double，计算输入）：公式中的输入`eps`，用于防止除0错误，数据类型为double，默认值为1e-6。
  * yOut（aclTensor*，计算输出）：公式中的`RmsNorm(x)`，支持1-8维度，shape需要与输入x1/x2一致，数据格式支持ND。不支持空Tensor。
    * Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
    * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  * rstdOut（aclTensor*，计算输出）：公式中的`Rms(x)`，数据类型支持FLOAT32、shape支持1-8维度，数据格式支持ND。不支持空Tensor。shape与x前几维保持一致，前几维表示不需要norm的维度。该输出在Atlas 推理系列产品上无效。
  * xOut（aclTensor*，计算输出）：公式中的`x`，shape支持1-8维度，shape需要与输入x1/x2一致，数据格式支持ND。不支持空Tensor。
    * Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
    * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  * workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
  * executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  返回161002（ACLNN_ERR_PARAM_INVALID）：输入和输出的数据类型不在支持的范围之内。
  ```

## aclnnAddRmsNorm

- **参数说明：**

  * workspace（void*，入参）：在Device侧申请的workspace内存地址。
  * workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnAddRmsNormGetWorkspaceSize获取。
  * executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
  * stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

- **返回值：**

  aclnnStatus：返回状态码。

## 约束与限制
- **功能维度**
  * 数据类型支持
    * Atlas 推理系列产品：x1、x2、gamma支持FLOAT32、FLOAT16。
    * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：x1、x2、gamma支持FLOAT32、FLOAT16、BFLOAT16。
    * rstd支持：FLOAT32。
  * 数据格式支持：ND。
  * Atlas 推理系列产品：x1、x2、gamma输入的尾轴长度必须大于等于 32 Bytes。
- **未支持类型说明**
  * DOUBLE：指令不支持DOUBLE。
  * 是否非连续tensor：不支持输入非连续，不支持数据非连续。
- **边界值场景说明**
  * 当输入是inf时，输出为inf。
  * 当输入是nan时，输出为nan。
