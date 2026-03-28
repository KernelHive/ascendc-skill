# aclnnBatchNorm

## 支持的产品型号
- Atlas 推理系列产品。
- Atlas 训练系列产品。
- Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件。
- Atlas A3 训练系列产品/Atlas A3 推理系列产品。

## 接口原型
每个算子分为两段式接口，必须先调用“aclnnBatchNormGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnBatchNorm”接口执行计算。

* `aclnnStatus aclnnBatchNormGetWorkspaceSize(const aclTensor *input, const aclTensor *weight, const aclTensor *bias, aclTensor *runningMean, aclTensor *runningVar, bool training, double momentum, double eps, aclTensor *output, aclTensor *saveMean, aclTensor *saveInvstd, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnBatchNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- 算子功能：
  对一个批次的数据做正则化处理，正则化之后生成的数据的统计结果为0均值、1标准差。

- 计算公式：

  $$
  y = \frac{(x - E[x])}{\sqrt{Var(x) + ε}} * γ + β
  $$
  E(x)表示均值，Var(x)表示方差，均需要在算子内部计算得到；ε表示一个极小的浮点数，防止分母为0的情况。

## aclnnBatchNormGetWorkspaceSize

- **参数说明：**
  
  - input（aclTensor*, 计算输入）：Device侧的aclTensor，支持非连续的Tensor，不支持空Tensor。支持的shape和格式有：2维（对应的格式为NC），3维（对应的格式为NCL），4维（对应的格式为NCHW），5维（对应的格式为NCDHW），6-8维（对应的格式为ND，其中第2维固定为channel轴）。
    - Atlas 推理系列产品、Atlas 训练系列产品：数据类型支持FLOAT、FLOAT16。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT、FLOAT16、BFLOAT16。

  - weight（aclTensor*, 计算输入）：可选参数，Device侧的aclTensor，权重Tensor。数据类型需要与input的数据类型一致，支持非连续的Tensor，不支持空Tensor。 数据格式为ND。shape为1维，长度与input入参中channel轴的长度相等。
    - Atlas 推理系列产品、Atlas 训练系列产品：数据类型支持FLOAT、FLOAT16。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT、FLOAT16、BFLOAT16。

  - bias（aclTensor*, 计算输入）：可选参数，Device侧的aclTensor，数据类型需要与input的数据类型一致，支持非连续的Tensor，不支持空Tensor。 数据格式为ND。shape为1维，长度与input入参中channel轴的长度相等。
    - Atlas 推理系列产品、Atlas 训练系列产品：数据类型支持FLOAT、FLOAT16。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT、FLOAT16、BFLOAT16。

  - runningMean（aclTensor*, 计算输入）：可选参数，Device侧的aclTensor，训练期间计算的平均值。数据类型需要与input的数据类型一致，支持非连续的Tensor，不支持空Tensor。 数据格式为ND。shape为1维，长度与input入参中channel轴的长度相等。
    - Atlas 推理系列产品、Atlas 训练系列产品：数据类型支持FLOAT、FLOAT16。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT、FLOAT16、BFLOAT16。

  - runningVar（aclTensor*, 计算输入）：可选参数，Device侧的aclTensor，训练期间计算的方差。数据类型需要与input的数据类型一致，数值为非负数，支持非连续的Tensor，不支持空Tensor。 数据格式为ND。shape为1维，长度与input入参中channel轴的长度相等。
    - Atlas 推理系列产品、Atlas 训练系列产品：数据类型支持FLOAT、FLOAT16。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT、FLOAT16、BFLOAT16。

  - training（bool, 计算输入）：Host侧的bool值，标记是否训练场景，true表示训练场景，false表示推理场景。

  - momentum（double, 计算输入）：Host侧的double值，动量参数。

  - eps（double, 计算输入）：Host侧的double值，添加到方差中的值，以避免出现除以零的情况。

  - output（aclTensor\*, 计算输出）：Device侧的aclTensor，数据类型需要与input的数据类型一致，支持非连续的Tensor，不支持空Tensor。shape与input入参的shape相同，支持的shape和格式有：2维（对应的格式为NC），3维（对应的格式为NCL），4维（对应的格式为NCHW），5维（对应的格式为NCDHW），6-8维（对应的格式为ND，其中第2维固定为channel轴）。
    - Atlas 推理系列产品、Atlas 训练系列产品：数据类型支持FLOAT，FLOAT16。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT，FLOAT16、BFLOAT16。

  - saveMean（aclTensor\*, 计算输出）：Device侧的aclTensor，保存的均值。数据类型需要与input的数据类型一致，支持非连续的Tensor，不支持空Tensor。 数据格式为ND。shape为1维，长度与input入参中channel轴的长度相等。
    - Atlas 推理系列产品、Atlas 训练系列产品：数据类型支持FLOAT，FLOAT16。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT，FLOAT16、BFLOAT16。

  - saveInvstd（aclTensor\*, 计算输出）：Device侧的aclTensor，保存的标准差的倒数。数据类型需要与input的数据类型一致，支持非连续的Tensor，不支持空Tensor。 数据格式为ND。shape为1维，长度与input入参中channel轴的长度相等。
    - Atlas 推理系列产品、Atlas 训练系列产品：数据类型支持FLOAT，FLOAT16。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT，FLOAT16、BFLOAT16。

  - workspaceSize（uint64_t\*, 出参）：返回需要在Device侧申请的workspace大小。
  
  - executor（aclOpExecutor\*\*, 出参）：返回op执行器，包含了算子计算流程。
  
- **返回值：**

  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的指针类型入参是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. input，output数据类型和数据格式不在支持的范围之内。
                                        2. input和output数据的shape不在支持的范围内。
  ```

## aclnnBatchNorm

- **参数说明：**
  * workspace（void\*, 入参）：在Device侧申请的workspace内存地址。
  * workspaceSize（uint64_t, 入参）：在Device侧申请的workspace大小，由第一段接口aclnnBatchNormGetWorkspaceSize获取。
  * executor（aclOpExecutor\*, 入参）：op执行器，包含了算子计算流程。
  * stream（aclrtStream, 入参）：指定执行任务的AscendCL Stream流。

- **返回值：**

  aclnnStatus：返回状态码。

## 约束与限制
无。

