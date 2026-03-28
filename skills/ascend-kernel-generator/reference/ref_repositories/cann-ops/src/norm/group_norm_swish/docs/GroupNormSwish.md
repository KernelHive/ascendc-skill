声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# aclnnGroupNormSwish

## 支持的产品型号

- Atlas A2 训练系列产品/Atlas 800I A2推理产品
- Atlas A3 训练系列产品/Atlas 800I A3推理产品

## 接口原型

每个算子分为两段式接口，必须先调用“aclnnGroupNormSwishGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGroupNormSwish”接口执行计算。

+ `aclnnStatus aclnnGroupNormSwishGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, const aclTensor *beta, int64_t numGroups, char *dataFormatOptional, double eps, bool activateSwish, double swishScale, const aclTensor *yOut, const aclTensor *meanOut, const aclTensor *rstdOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
+ `aclnnStatus aclnnGroupNormSwish(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

+ 接口功能：计算输入x的组归一化结果out，均值meanOut，标准差的倒数rstdOut，以及swish的输出。
+ 计算公式：
  - **GroupNorm:**
  记 $E[x] = \bar{x}$代表$x$的均值，$Var[x] = \frac{1}{n} * \sum_{i=1}^n(x_i - E[x])^2$代表$x$的方差，则
  $$
  \left\{
  \begin{array} {rcl}
  yOut& &= \frac{x - E[x]}{\sqrt{Var[x] + eps}} * \gamma + \beta \\
  meanOut& &= E[x]\\
  rstdOut& &= \frac{1}{\sqrt{Var[x] + eps}}\\
  \end{array}
  \right.
  $$
  - **Swish:**
  $$
  yOut = \frac{x}{1+e^{-scale * x}}
  $$
      当activateSwish为True时，会计算Swish， 此时swish计算公式的x为GroupNorm公式得到的out。

## aclnnGroupNormSwishGetWorkspaceSize

- **参数说明：**

  * x（aclTensor*， 计算输入）：表示待组归一化的目标张量，`yOut`计算公式中的$x$，维度需大于一维，要求x第0维和第1维大于0，第1维要求能被group整除，Device侧的aclTensor。数据类型支持FLOAT32、FLOAT16、BFLOAT16，数据格式支持ND，支持非连续的Tensor。
  * gamma（aclTensor*， 计算输入）：表示组归一化中的 gamma 参数，`yOut`计算公式中的$\gamma$，维度为一维，元素数量需与输入$x$的第1维度相同，gamma与beta的数据类型必须保持一致，且数据类型与x相同或者为FLOAT，Device侧的aclTensor。数据类型支持FLOAT32、FLOAT16、BFLOAT16，数据格式支持ND，支持非连续的Tensor。
  * beta（aclTensor*， 计算输入）：表示组归一化中的 beta 参数，`yOut`计算公式中的$\beta$，维度为一维，元素数量需与输入$x$的第1维度相同，gamma与beta的数据类型必须保持一致，且数据类型与x相同或者为FLOAT，Device侧的aclTensor。数据类型支持FLOAT32、FLOAT16、BFLOAT16，数据格式支持ND，支持非连续的Tensor。
  * numGroups（int64\_t， 计算输入）： 表示将输入$x$的第1维度分为group组，值范围大于0，Host侧的整数。
  * dataFormatOptional（char*，计算输入）： 表示数据格式，当前版本只支持输入NCHW，Host侧的字符类型。
  * eps（double， 计算输入）： 用于防止产生除0的偏移，`yOut`和`rstdOut`计算公式中的$eps$值，值范围大于0，Host侧的DOUBLE类型。
  * activateSwish（bool， 计算输入）： 表示是否支持swish计算。如果设置为true，则表示groupnorm计算后继续swish计算，Host侧的BOOL类型。
  * swishScale（double， 计算输入）： 表示Swish计算时的$scale$值，Host侧的DOUBLE类型。
  * yOut（aclTensor*， 计算输出）： 表示组归一化结果，数据类型和shape与$x$相同，Device 侧的aclTensor。数据类型支持FLOAT32、FLOAT16、BFLOAT16，数据格式支持ND，支持非连续的Tensor。
  * meanOut（aclTensor*， 计算输出）： 表示x分组后的均值，数据类型与$gamma$相同，shape为`(N， numGroups)`，其中`N`表示$x$第0维度的大小，`numGroups`为计算输入，表示将输入$x$的第1维度分为group组，Device 侧的aclTensor。数据类型支持FLOAT32、FLOAT16、BFLOAT16，数据格式支持ND，支持非连续的Tensor。
  * rstdOut（aclTensor*， 计算输出）： 表示x分组后的标准差的倒数，数据类型与$gamma$相同，shape为`(N， numGroups)`，其中`N`表示$x$第0维度的大小，`numGroups`为计算输入，表示将输入$x$的第1维度分为group组，Device 侧的aclTensor。数据类型支持FLOAT32、FLOAT16、BFLOAT16，数据格式支持ND，支持非连续的Tensor。
  * workspaceSize（uint64_t\*， 出参）： 返回需要在Device侧申请的workspace大小。
  * executor（aclOpExecutor **， 出参）： 返回op执行器，包含算子计算流程。

- **返回值：**

  aclnnStatus： 返回状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

```
第一段接口完成入参校验，出现以下场景时报错：
161001 ACLNN_ERR_PARAM_NULLPTR：1. 传入的x、out、meanOut、rstdOut是空指针时。
161002 ACLNN_ERR_PARAM_INVALID：1. x、gamma、beta、out、meanOut、rstdOut数据类型不在支持的范围之内。
```

## aclnnGroupNormSwish

- **参数说明：**

  * workspace(void*, 入参)：在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnGroupNormSwishGetWorkspaceSize获取。
  * executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。
  * stream(aclrtStream, 入参)：指定执行任务的AscendCL Stream流。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束与限制
无

## 调用示例

详见[GroupNormSwish自定义算子样例说明算子调用章节](../README.md#算子调用)
