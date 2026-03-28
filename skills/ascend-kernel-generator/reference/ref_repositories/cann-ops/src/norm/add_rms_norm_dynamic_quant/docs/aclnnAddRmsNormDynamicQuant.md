# aclnnAddRmsNormDynamicQuant

## 支持的产品型号

- Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件。
- Atlas A3 训练系列产品/Atlas A3 推理系列产品。

## 接口原型

每个算子分为两段式接口，必须先调用`aclnnAddRmsNormDynamicQuantGetWorkspaceSize`接口获取入参并根据计算流程所需workspace大小，再调用`aclnnAddRmsNormDynamicQuant`接口执行计算。

- `aclnnStatus aclnnAddRmsNormDynamicQuantGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *gamma, const aclTensor *smoothScale1Optional, const aclTensor *smoothScale2Optional, double epsilon, aclTensor *y1Out, aclTensor *y2Out, aclTensor *xOut, aclTensor *scale1Out, aclTensor *scale2Out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnAddRmsNormDynamicQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- 算子功能：RmsNorm算子是大模型常用的归一化操作，相比LayerNorm算子，其去掉了减去均值的部分。DynamicQuant算子则是为输入张量进行对称动态量化的算子。AddRmsNormDynamicQuant算子将 RmsNorm前的Add算子和RmsNorm归一化输出给到的1个或2个DynamicQuant算子融合起来，减少搬入搬出操作。
- 计算公式：

  $$
  x=x_{1}+x_{2}
  $$

  $$
  y = \operatorname{RmsNorm}(x)=\frac{x}{\operatorname{Rms}(\mathbf{x})}\cdot gamma, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  - 若smoothScale1Optional和smoothScale2Optional均不输入，则y2Out和scale2Out输出无实际意义。计算过程如下所示：

    $$
     scale1Out=row\_max(abs(y))/127
    $$

    $$
     y1Out=round(y/scalOut)
    $$

  - 若仅输入smoothScale1Optional，则y2Out和scale2Out输出无实际意义。计算过程如下所示：
    
    $$
      input = y\cdot smoothScale1Optional
    $$
    $$
     scale1Out=row\_max(abs(input))/127
    $$

    $$
     y1Out=round(input/scale1Out)
    $$

  - 若smoothScale1Optional和smoothScale2Optional均输入，则算子的五个输出均为有效输出。计算过程如下所示：
   
    $$
      input1 = y\cdot smoothScale1Optional
    $$
    $$
      input2 = y\cdot smoothScale2Optional
    $$
    $$
     scale1Out=row\_max(abs(input1))/127
    $$
    $$
     scale2Out=row\_max(abs(input2))/127
    $$
    $$
     y1Out=round(input1/scale1Out)
    $$
    $$
     y2Out=round(input2/scale2Out)
    $$

  其中row\_max代表每行求最大值。

## aclnnAddRmsNormDynamicQuantGetWorkspaceSize

- **参数说明：**

  - x1（aclTensor*，计算输入）：公式中的输入`x1`，表示标准化过程中的源数据张量，Device侧的aclTensor。shape支持2-8维，数据类型支持FLOAT16、BFLOAT16。数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
  - x2（aclTensor*，计算输入）：公式中的输入`x2`，表示标准化过程中的源数据张量，Device侧的aclTensor。shape和数据类型需要与x1保持一致。数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
  - gamma（aclTensor*，计算输入）：公式中的输入`gamma`，表示标准化过程中的权重张量，Device侧的aclTensor。shape支持1维，shape需要与x1最后一维一致，数据类型需要与x1保持一致。数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
  - smoothScale1Optional（aclTensor*，计算输入）：公式中的输入`smoothScale1Optional`，表示量化过程中得到y1使用的smoothScale张量，Device侧的aclTensor。可选参数，支持传入空指针。shape和数据类型需要与gamma保持一致。数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
  - smoothScale2Optional（aclTensor*，计算输入）：公式中的输入`smoothScale2Optional`，表示量化过程中得到y2使用的smoothScale张量，Device侧的aclTensor。可选参数，支持传入空指针。必须与smoothScale1Optional配套使用。shape和数据类型需要与gamma保持一致。数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
  - epsilon（double，计算输入）：公式中的输入`epsilon`，用于防止除0错误，数据类型为double，建议传入较小正数，如1e-6。
  - y1Out（aclTensor*，计算输出）：公式中的输出`y1Out`，表示量化输出Tensor，Device侧的aclTensor。shape需要与输入x1/x2一致，或者是二维并且第一维等于x1除最后一维的维度乘积，第二维等于x1的最后一维，数据类型支持INT8，数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
  - y2Out（aclTensor*，计算输出）：公式中的输出`y2Out`，表示量化输出Tensor，Device侧的aclTensor。当smoothScale2Optional不存在时，此输出无意义。shape需要与y1Out一致，数据类型支持INT8，数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
  - xOut（aclTensor*，计算输出）：表示x1和x2的和，公式中的输出`x`，Device侧的aclTensor。shape和数据类型需要与输入x1/x2一致，数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
  - scale1Out（aclTensor*，计算输出）：公式中的输出`scale1Out`，第一路量化的输出，Device侧的aclTensor。shape需要与输入x1除最后一维后的shape一致，或者与x1除最后一维的乘积一致，数据类型支持FLOAT32。数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
  - scale2Out（aclTensor*，计算输出）：公式中的输出`scale2Out`，第二路量化的输出，Device侧的aclTensor。当smoothScale2Optional不存在时，此输出无意义。shape需要与scale1Out一致，数据类型支持FLOAT32。数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus: 返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：

  返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  返回161002（ACLNN_ERR_PARAM_INVALID）：输入和输出的数据类型不在支持的范围之内。
  返回561002（ACLNN_ERR_INNER_TILING_ERROR）：1. 输入smoothScale2Optional，而没有输入smoothScale1Optional。
                                             2. 输入/输出的shape关系不符合预期。
  ```

## aclnnAddRmsNormDynamicQuant

- **参数说明：**

  * workspace（void*，入参）：在Device侧申请的workspace内存地址。
  * workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnAddRmsNormDynamicQuantGetWorkspaceSize获取。
  * executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
  * stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

- **返回值：**

  aclnnStatus：返回状态码。

## 约束与限制
**各产品型号支持数据类型说明**

  | x1 数据类型 | x2 数据类型 | gamma 数据类型 | smoothScale1Optional 数据类型 | smoothScale2Optional 数据类型 | y1Out 数据类型 | y2Out 数据类型 | scale1Out 数据类型 | scale2Out 数据类型 |
  | ----------- | ----------- | -------------- | ----------------------------- | ----------------------------- | -------------- | -------------- | ------------------ | ------------------ |
  | float16     | float16     | float16        | float16                       | float16                       | int8           | int8           | float32            | float32            |
  | bfloat16    | bfloat16    | bfloat16       | bfloat16                      | bfloat16                      | int8           | int8           | float32            | float32            |
