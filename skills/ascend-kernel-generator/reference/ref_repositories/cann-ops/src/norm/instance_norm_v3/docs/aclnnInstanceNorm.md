# aclnnInstanceNorm

## 支持的产品型号

- Atlas 推理系列产品。

## 接口原型

每个算子分为两段式接口，必须先调用`aclnnInstanceNormGetWorkspaceSize`接口获取入参并根据计算流程所需workspace大小，再调用`aclnnInstanceNorm`接口执行计算。

*  `aclnnStatus aclnnInstanceNormGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, const aclTensor *beta, const char *dataFormat, double eps, aclTensor *y, aclTensor *mean, aclTensor *variance, uint64_t *workspaceSize, aclOpExecutor **executor)`
*  `aclnnStatus aclnnInstanceNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述
- **算子功能**：计算InstanceNorm功能。
- **计算公式**：

  $$
  y = {{x-E(x)}\over\sqrt {Var(x)+eps}} * \gamma + \beta
  $$


## aclnnInstanceNormGetWorkspaceSize

- **参数说明：**

  * x（aclTensor\*，计算输入）：公式中的`x`。Device 侧的aclTensor，tensor维度为4维。数据类型支持FLOAT、FLOAT16。支持非连续的Tensor，不支持空Tensor。数据格式支持ND，实际数据格式由参数dataFormat决定。
  * gamma（aclTensor\*，计算输入）：公式中的`gamma`。Device 侧的aclTensor，tensor维度为1维，且 shape 和输入 `x` 的C轴一致。数据类型支持FLOAT、FLOAT16，和输入 `x` 保持一致。支持非连续的Tensor，不支持空Tensor。数据格式支持ND。
  * beta（aclTensor\*，计算输入）：公式中的`beta`。Device 侧的aclTensor，tensor维度为1维，且 shape 和输入 `x` 的C轴一致。数据类型支持FLOAT、FLOAT16，和输入 `x` 保持一致。支持非连续的Tensor，不支持空Tensor。数据格式支持ND。
  * dataFormat（char\*，计算输入）：算子输入Tensor的实际数据排布，可以是"NHCW"或"NCHW"。
  * eps（double\*，计算输入）：对应InstanceNorm计算公式中的 eps，添加到分母中的值，以确保数值稳定。
  * y（aclTensor\*，计算输出）：表示InstanceNorm的结果输出`y`。Device 侧的aclTensor，数据类型支持FLOAT、FLOAT16，且数据类型与`x`一致。 shape需要与 `x`一致，数据格式支持ND。
  * mean（aclTensor\*，计算输出）：公式中`E(x)`的计算结果。tensor维度为4维，shape与输入x满足broadcast关系（前2维的shape和输入x前2维的shape相同，前2维表示不需要norm的维度，其余维度大小为1）。数据类型支持FLOAT、FLOAT16，且和输入 `x` 保持一致。数据格式支持ND。
  * variance（aclTensor\*，计算输出）：公式中`Var(x)`的计算结果。tensor维度为4维，shape与输入x满足broadcast关系（前2维的shape和输入x前2维的shape相同，前2维表示不需要norm的维度，其余维度大小为1）。数据类型支持FLOAT、FLOAT16，且和输入 `x` 保持一致。数据格式支持ND。
  * workspaceSize（uint64_t\*，出参）：返回用户需要在Device 侧申请的workspace大小。
  * executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的 x，gamma，beta，y，mean，variance 是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. 传入的 x，gamma，beta，y，mean，variance 的数据类型不在支持范围内。
                                        2. x的维度不为 4 或 gamma/beta的维度非 1。
                                        3. gamma，beta的 shape 和 x 的 C 轴不一致。
                                        4. 非支持的产品型号。
                                        5. dataFormat 没有设置为 'NCHW' 或 'NHWC'。
  返回561103（ACLNN_ERR_INNER_NULLPTR）：1. aclnn 接口中间计算结果出现 nullptr。
                                        2. C轴 或 H*W 长度小于32B
  返回561101（ACLNN_ERR_INNER_CREATE_EXECUTOR）：1. API内部创建aclOpExecutor失败。
  ```

## aclnnInstanceNorm
- **参数说明：**
  * workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  * workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnInstanceNormGetWorkspaceSize获取。
  * executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  * stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

- **返回值：**

  aclnnStatus：返回状态码。

## 约束与限制
- **功能维度**
  * 数据类型支持
    * x，gamma，beta，y，mean，variance 支持：FLOAT32、FLOAT16。
  * 数据格式支持：ND。
  * 仅支持310P
  * x，y的shape要求4维，gamma/beta的维度要求1维，且和x，y的C轴一致。
  * x，y的 H\*W 大小需要大于 32Bytes，C轴需要大于 32Bytes
  * 参数 dataFormat 仅支持 "NHWC" 和 "NCHW"
- **边界值场景说明**
  * 当输入是inf时，输出为inf。
  * 当输入是nan时，输出为nan。
