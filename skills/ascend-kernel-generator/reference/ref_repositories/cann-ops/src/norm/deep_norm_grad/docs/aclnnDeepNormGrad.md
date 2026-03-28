# aclnnDeepNormGrad

## 支持的产品型号

- Atlas 推理系列产品。
- Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件。
- Atlas A3 训练系列产品/Atlas A3 推理系列产品。

## 接口原型
每个算子分为两段式接口，必须先调用“aclnnDeepNormGradGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnDeepNormGrad”接口执行计算。
*  `aclnnStatus aclnnDeepNormGradGetWorkspaceSize(const aclTensor *dy, const aclTensor *x, const aclTensor *gx, const aclTensor *gamma, const aclTensor *mean, const aclTensor *rstd, double alpha, aclTensor *dxOut, aclTensor *dgxOut, aclTensor *dbetaOut, aclTensor *dgammaOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
*  `aclnnStatus aclnnDeepNormGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述
- **算子功能**：DeepNorm算子的反向计算

- **计算公式**：

    $$
    d_{gx_i} = tmpone_i * rstd + \frac{2}{D} * d_{var} * tmptwo_i + {\frac{1}{D}} * d_{mean}
    $$

    $$
    d_{x_i} = alpha * {gx}_i
    $$

    $$
    d_{beta} = \sum_{i=1}^{N} d_{y_i}
    $$

    $$
    d_{gamma} =  \sum_{i=1}^{N} d_{y_i} * rstd * {tmptwo}_i
    $$

    其中：
    $$
    tmpone_i = d_{y_i} * gamma
    $$

    $$
    tmptwo_i = alpha * x_i + {gx}_i - mean
    $$

    $$
    d_{var} = \sum_{i=1}^{N} (-0.5) * {tmpone}_i * {tmptwo}_i * {rstd}^3
    $$

    $$
    d_{mean} = \sum_{i=1}^{N} (-1) * {tmpone}_i * rstd
    $$

## aclnnDeepNormGradGetWorkspaceSize

- **参数说明：**
  * dy（aclTensor\*，计算输入）：公式中的输入$d_y$，主要的grad输入。Device侧的aclTensor，shape支持2维-8维，数据格式支持ND，不支持非连续输入，不支持空tensor。
    * Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
    * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  * x（aclTensor\*，计算输入）：公式中的输入`x`，为正向融合算子的输入x。Device侧的aclTensor，shape支持2维-8维，数据格式支持ND，不支持非连续输入，不支持空tensor。
    * Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
    * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  * gx（aclTensor\*，计算输入）：公式中的输入`gx`，为正向融合算子的输入gx。Device侧的aclTensor，shape支持2维-8维，数据格式支持ND，不支持非连续输入，不支持空tensor。
    * Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
    * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  * gamma（aclTensor\*，计算输入）：公式中的输入`gamma`，shape支持1维-7维，数据格式支持ND，不支持非连续输入，不支持空tensor。
    * Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
    * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  * mean（aclTensor\*，计算输入）：公式中的输入`mean`，表示正向输入x、gx之和的均值。数据类型支持FLOAT32，shape支持2维-8维，数据格式支持ND，不支持非连续输入，不支持空tensor。
  * rstd（aclTensor\*，计算输入）：公式中的输入`rstd`，表示正向输入x、gx之和的rstd。输入数据类型支持FLOAT32，shape支持2维-8维，数据格式支持ND，不支持非连续输入，不支持空tensor。
  * alpha(double，计算输入)：公式中的输入`alpha`，含义与deepnorm正向输入alpha相同，deepnorm输入x维度的乘数。
  * dxOut（aclTensor\*，计算输出）：公式中的输出$d_x$，shape支持2维-8维，数据格式支持ND。不支持空tensor。
    * Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
    * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  * dgxOut（aclTensor\*，计算输出）：公式中的输出$d_{gx}$，shape支持2维-8维，数据格式支持ND。不支持空tensor。
    * Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
    * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  * dbetaOut（aclTensor\*，计算输出）：公式中的输出$d_{beta}$，数据类型支持FLOAT32，shape支持1维-7维，数据格式支持ND。不支持空tensor。
  * dgammaOut（aclTensor\*，计算输出）：公式中的输出$d_{gamma}$，数据类型支持FLOAT32，shape支持1维-7维，数据格式支持ND。不支持空tensor。
  * workspaceSize（uint64_t\*，出参）：返回需要在Device侧申请的workspace大小。
  * executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. 输入和输出的数据类型不在支持的范围之内.
                                        2. 输入和输出的shape不匹配或者不在支持的维度范围内。
  ```

## aclnnDeepNormGrad
- **参数说明：**
  * workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  * workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnDeepNormGradGetWorkspaceSize获取。
  * executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  * stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

- **返回值：**

  aclnnStatus：返回状态码。

## 约束与限制
- **功能维度**
  * 数据类型支持
    * Atlas 推理系列产品：dy、x、gx、gamma支持FLOAT32、FLOAT16。
    * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：dy、x、gx、gamma支持FLOAT32、FLOAT16、BFLOAT16。
    * rstd、mean支持：FLOAT32。
  * 数据格式支持：ND。
- **未支持类型说明**
  * DOUBLE：指令不支持DOUBLE。
  * 是否支持空tensor：不支持空进空出。
  * 是否支持非连续tensor：不支持输入非连续。
- **边界值场景说明**
  * 当输入是inf时，输出为inf。
  * 当输入是nan时，输出为nan。
