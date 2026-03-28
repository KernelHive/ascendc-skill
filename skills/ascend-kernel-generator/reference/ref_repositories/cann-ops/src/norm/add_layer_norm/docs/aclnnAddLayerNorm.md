# aclnnAddLayerNorm

## 支持的产品型号

- Atlas 推理系列产品。
- Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件。
- Atlas A3 训练系列产品/Atlas A3 推理系列产品。

## 接口原型

每个算子分为两段式接口，必须先调用`aclnnAddLayerNormGetWorkspaceSize`接口获取入参并根据计算流程所需workspace大小，再调用`aclnnAddLayerNorm`接口执行计算。

*  `aclnnStatus aclnnAddLayerNormGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *gamma, const aclTensor *beta, const aclTensor *bias, double epsilon, bool additionalOut, const aclTensor *yOut, const aclTensor *meanOut, const aclTensor *rstdOut, const aclTensor *xOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
*  `aclnnStatus aclnnAddLayerNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述
- **算子功能**：实现AddLayerNorm功能。
- **计算公式**：

  $$
  x = x1 + x2 + bias
  $$

  $$
  y = {{x-\bar{x}}\over\sqrt {Var(x)+eps}} * \gamma + \beta
  $$


## aclnnAddLayerNormGetWorkspaceSize

- **参数说明：**

  * x1（aclTensor\*，计算输入）：公式中的输入`x1`，表示AddLayerNorm中加法计算的输入，将会在算子内做(x1 + x2 + bias)的计算并对计算结果做层归一化；是Device侧的aclTensor，shape支持1-8维度，不支持输入的某一维的值为0，数据格式支持ND。不支持空Tensor。
    * Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
    * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  * x2（aclTensor\*，计算输入）：公式中的输入`x2`，表示AddLayerNorm中加法计算的输入，将会在算子内做(x1 + x2 + bias)的计算并对计算结果做层归一化；是Device侧的aclTensor，shape需要与x1一致，数据格式支持ND。不支持空Tensor。
    * Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
    * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  * beta（aclTensor\*，计算输入）：对应LayerNorm计算公式中的`beta`，表示层归一化中的beta参数；是Device侧的aclTensor，shape支持1-8维度，与x1需要norm的维度的维度值相同，数据格式支持ND。不支持空Tensor。
    * Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
    * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  * gamma（aclTensor\*，计算输入）：对应LayerNorm计算公式中的`gamma`，表示层归一化中的gamma参数；是Device侧的aclTensor，shape支持1-8维度，与x1需要norm的维度的维度值相同，数据格式支持ND。不支持空Tensor。
    * Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
    * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  * bias（aclTensor\*，计算输入）：可选输入参数，表示AddLayerNorm中加法计算的输入，将会在算子内做(x1 + x2 + bias)的计算并对计算结果做层归一化；shape可以和gamma/beta或是和x1/x2一致，是Device侧的aclTensor，shape支持1-8维度，数据格式支持ND。不支持空Tensor。
    * Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
    * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  * epsilon（double\*，计算输入）：公式中的输入`eps`，添加到分母中的值，以确保数值稳定；host侧的aclScalar，数据类型为double，仅支持取值1e-5。
  * additionalOut（bool\*，计算输入）：表示是否开启x=x1+x2+bias的输出，host侧的aclScalar，数据类型为bool。
  * meanOut（aclTensor\*，计算输出）：输出LayerNorm计算过程中(x1 + x2 + bias)的结果的均值，Device侧的aclTensor，数据类型为FLOAT32，shape需要与x1满足broadcast关系（前几维的维度和x1前几维的维度相同，前几维指x1的维度减去gamma的维度，表示不需要norm的维度），数据格式支持ND。不支持空Tensor。计算逻辑：mean = np.mean(x1 + x2 + bias)。
    * Atlas 推理系列产品：该输出在本产品无效。
  * rstdOut（aclTensor\*，计算输出）：输出LayerNorm计算过程中rstd的结果，Device侧的aclTensor，数据类型为FLOAT32，shape需要与x1满足broadcast关系（前几维的维度和x1前几维的维度相同），数据格式支持ND。不支持空Tensor。计算逻辑：rstd = np.power((np.var(x1 + x2 + bias) + epsilon), (-0.5))。
    * Atlas 推理系列产品：该输出在本产品无效。
  * yOut（aclTensor\*，计算输出）：表示LayerNorm的结果输出y，Device侧的aclTensor，shape需要与输入x1/x2一致，数据格式支持ND。不支持空Tensor。
    * Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
    * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  * xOut（aclTensor\*，计算输出）：表示LayerNorm的结果输出x，Device侧的aclTensor，shape需要与输入x1/x2一致，数据格式支持ND。不支持空Tensor。
    * Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
    * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  * workspaceSize（uint64_t\*，出参）：返回需要在Device侧申请的workspace大小。
  * executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  返回161002（ACLNN_ERR_PARAM_INVALID）：输入和输出的数据类型不在支持的范围之内。
  返回561002（ACLNN_ERR_INNER_TILING_ERROR）：算子tiling过程中触发主动拦截， 包括如下情况：
    1. tiling阶段（x1、x2、gamma、beta、y、mean、rstd、x）的shape获取失败。
    2. （x1、gamma）的shape维数大于8或小于0。
    3. （x1、x2、y、mean、rstd、x）的维数不一致。
    4. x1的维数小于gamma。
    5. （x1、gamma、mean）的任意一个维度等于0， 且（y、mean、rstd、x）的shape不全是任意一个维度等于0。
    6. （x1、x2、y、x）的shape不是完全相同的shape。
    7. （gamma、beta）的shape不是完全相同的shape。
    8. （mean、rstd）的shape不是完全相同的shape。
    9. gamma的维度和x的需要作norm的维度不相同，或者是mean的维度和x的不需要norm的维度不相同，或是mean的需要norm的维度不为1。

  ```

## aclnnAddLayerNorm
- **参数说明：**
  * workspace（void\*，入参）：在Device侧申请的workspace内存返回需要在Device侧。
  * workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnAddLayerNormGetWorkspaceSize获取。
  * executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  * stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

- **返回值：**

  aclnnStatus：返回状态码。

## 约束与限制
- **功能维度**
  * 数据类型支持
    * Atlas 推理系列产品：x1、x2、beta、gamma、bias支持FLOAT32、FLOAT16。
    * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：x1、x2、beta、gamma、bias支持FLOAT32、FLOAT16、BFLOAT16。
    * rstd、mean支持：FLOAT32。
  * 数据格式支持：ND。
  * Atlas 推理系列产品：x1、x2、beta、gamma、bias五个输入的尾轴长度必须大于等于 32 Bytes。
- **未支持类型说明**
  * DOUBLE：不支持DOUBLE。
  * 是否支持空tensor：不支持空进空出。
  * 是否非连续tensor：不支持输入非连续。
- **边界值场景说明**
  * 当输入是inf时，输出为inf。
  * 当输入是nan时，输出为nan。
- **各产品支持数据类型说明**
  - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：
    | x1 数据类型 | x2 数据类型 | gamma 数据类型 | beta 数据类型 | bias 数据类型 | y 数据类型 | mean 数据类型 | rstd 数据类型 | x 数据类型 |
    | -------- | -------- | ------------- | ------------- | ----------- | --------- | --------- | --------- | :-------- |
    | float32  | float16  | float32  | float32  | float32  | float32  | float32  | float32  | float32  |
    | float32  | bfloat16 | float32  | float32  | float32  | float32  | float32  | float32  | float32  |
    | float16  | float32  | float32  | float32  | float32  | float32  | float32  | float32  | float32  |
    | bfloat16 | float32  | float32  | float32  | float32  | float32  | float32  | float32  | float32  |
    | float16  | float16  | float32  | float32  | float16  | float16  | float32  | float32  | float16  |
    | bfloat16 | bfloat16 | float32  | float32  | bfloat16 | bfloat16 | float32  | float32  | bfloat16 |
    | float16  | float16  | float16  | float16  | float16  | float16  | float32  | float32  | float16  |
    | bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 | float32  | float32  | bfloat16 |
    | float32  | float32  | float32  | float32  | float32  | float32  | float32  | float32  | float32  |
  - Atlas 推理系列产品：
    | x1 数据类型 | x2 数据类型 | gamma 数据类型 | beta 数据类型 | bias 数据类型 | y 数据类型 | mean 数据类型 | rstd 数据类型 | x 数据类型 |
    | -------- | -------- | ------------- | ------------- | ----------- | --------- | --------- | --------- | :-------- |
    | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 |
    | float16 | float16 | float16 | float16 | float16 | float16 | float32 | float32 | float16 |
