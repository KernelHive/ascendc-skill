# aclnnAddRmsNormQuant

## 支持的产品型号

- Atlas 推理系列产品。
- Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件。
- Atlas A3 训练系列产品/Atlas A3 推理系列产品。

## 接口原型

每个算子分为两段式接口，必须先调用`aclnnAddRmsNormQuantGetWorkspaceSize`接口获取入参并根据计算流程所需workspace大小，再调用`aclnnAddRmsNormQuant`接口执行计算。

* `aclnnStatus aclnnAddRmsNormQuantGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *gamma, const aclTensor *scales1, const aclTensor *scales2Optional, const aclTensor *zeroPoints1Optional, const aclTensor *zeroPoints2Optional, int64_t axis, double epsilon, bool divMode, aclTensor *y1Out, aclTensor *y2Out, aclTensor *xOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnAddRmsNormQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- 算子功能：RmsNorm算子是大模型常用的标准化操作，相比LayerNorm算子，其去掉了减去均值的部分。AddRmsNormQuant算子将RmsNorm前的Add算子以及RmsNorm后的Quantize算子融合起来，减少搬入搬出操作。
- 计算公式：

  $$
  x_i={x1}_{i1}+{x2}_{i2}
  $$

  $$
  y_i=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} g_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}
  $$

  - divMod为True时：

    $$
    y1=round((y/scales1)+zero\_points1)
    $$

    $$
    y2=round((y/scales2)+zero\_points2)
    $$

  - divMod为False时：

    $$
    y1=round((y*scales1)+zero\_points1)
    $$

    $$
    y2=round((y*scales2)+zero\_points2)
    $$

## aclnnAddRmsNormQuantGetWorkspaceSize

- **参数说明：**

  - x1（aclTensor*，计算输入）：表示标准化过程中的源数据张量，公式中的`x1`，Device侧的aclTensor。shape支持1-8维。数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
    - Atlas 推理系列产品：数据类型支持FLOAT16。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT16、BFLOAT16。
  - x2（aclTensor*，计算输入）：表示标准化过程中的源数据张量，公式中的`x2`，Device侧的aclTensor。shape支持1-8维，shape和数据类型需要与x1保持一致。数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
    - Atlas 推理系列产品：数据类型支持FLOAT16。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT16、BFLOAT16。
  - gamma（aclTensor*，计算输入）：表示标准化过程中的权重张量，公式中的`g`，Device侧的aclTensor。shape支持1-8维，shape需要与x1需要Norm的维度保持一致，数据类型需要与x1保持一致。数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
    - Atlas 推理系列产品：数据类型支持FLOAT16。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT16、BFLOAT16。
  - scales1（aclTensor*，计算输入）：表示量化过程中得到y1进行的scales张量，公式中的`scales1`，Device侧的aclTensor。shape需要与gamma保持一致。数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
    - Atlas 推理系列产品：数据类型支持FLOAT32。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32、BFLOAT16。
  - scales2Optional（aclTensor*，计算输入）：表示量化过程中得到y2进行的scales张量，公式中的`scales2`，Device侧的aclTensor。可选参数，支持传入空指针。shape需要与gamma保持一致，数据类型需要与scales1保持一致。数据格式支持ND，支持非连续的Tensor，不支持空Tensor。预留参数，实际未使用。
    - Atlas 推理系列产品：数据类型支持FLOAT32。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32、BFLOAT16。
  - zeroPoints1Optional（aclTensor*，计算输入）：表示量化过程中得到y1进行的offset张量，公式中的`zero_points1`，Device侧的aclTensor。可选参数，支持传入空指针。shape需要与gamma保持一致。数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
    - Atlas 推理系列产品：数据类型支持INT32。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持INT32、BFLOAT16。
  - zeroPoints2Optional（aclTensor*，计算输入）：表示量化过程中得到y2进行的offset张量，公式中的`zero_points2`，Device侧的aclTensor。可选参数，支持传入空指针。shape需要与gamma保持一致，数据类型需要与zeroPoints1Optional保持一致。数据格式支持ND，支持非连续的Tensor，不支持空Tensor。预留参数，实际未使用。
    - Atlas 推理系列产品：数据类型支持INT32。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持INT32、BFLOAT16。
  - axis（int64_t，计算输入）：Host侧的整型，表示需要进行量化的elewise轴，其他的轴做broadcast，指定的轴不能超过输入x的维度数。当前仅支持-1，传其他值均不生效。
  - epsilon（double，计算输入）： 公式中的输入eps，用于防止除0错误，数据类型为double。建议传较小的正数。
  - divMode（bool，计算输入）： 公式中决定量化公式是否使用除法的参数，数据类型为bool，当前仅支持True，传其他值均不生效。
  - y1Out（aclTensor*，计算输出）：表示量化输出Tensor，公式中的`y1`，Device侧的aclTensor。shape支持1-8维度，shape需要与输入x1/x2一致，数据类型支持INT8，数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
  - y2Out（aclTensor*，计算输出）：表示量化输出Tensor，公式中的`y2`，Device侧的aclTensor。shape支持1-8维度，shape需要与输入x1/x2一致，数据类型支持INT8，数据格式支持ND，支持非连续的Tensor，不支持空Tensor。预留参数，实际未使用，输出为随机值。
  - xOut（aclTensor*，计算输出）：表示x1和x2的和，公式中的`x`，Device侧的aclTensor。shape支持1-8维度，shape和数据类型需要与输入x1/x2一致，数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
    - Atlas 推理系列产品：数据类型支持FLOAT16。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT16、BFLOAT16。
  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：

  返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  返回161002（ACLNN_ERR_PARAM_INVALID）：输入和输出的数据类型不在支持的范围之内。
  ```

## aclnnAddRmsNormQuant

- **参数说明：**

  * workspace（void*，入参）：在Device侧申请的workspace内存地址。
  * workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnAddRmsNormQuantGetWorkspaceSize获取。
  * executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
  * stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

- **返回值：**

  aclnnStatus：返回状态码。

## 约束与限制

**各产品型号支持数据类型说明**
  - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：

    | x1 数据类型 | x2 数据类型 | gamma 数据类型 | scales1 数据类型 | scales2Optional 数据类型 | zeroPoints1Optional 数据类型 | zeroPoints2Optional 数据类型 | y1Out 数据类型 | y2Out 数据类型 | xOut 数据类型 |
    | - | - | - | - | - | - | - | - | - | - |
    | float16 | float16 | float16 | float32 | float32 | int32 | int32 | int8 | int8 | float16 |
    | bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 | int8 | int8 | bfloat16 |

  - Atlas 推理系列产品：

    | x1 数据类型 | x2 数据类型 | gamma 数据类型 | scales1 数据类型 | scales2Optional 数据类型 | zeroPoints1Optional 数据类型 | zeroPoints2Optional 数据类型 | y1Out 数据类型 | y2Out 数据类型 | xOut 数据类型 |
    | - | - | - | - | - | - | - | - | - | - |
    | float16 | float16 | float16 | float32 | float32 | int32 | int32 | int8 | int8 | float16 |
