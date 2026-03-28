# aclnnLayerNorm&aclnnLayerNormWithImplMode

## 支持的产品型号

- Atlas 训练系列产品。
- Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件。
- Atlas A3 训练系列产品/Atlas A3 推理系列产品。
- Atlas 200I/500 A2 推理产品。

## 接口原型

每个算子分为两段式接口，必须先调用“aclnnLayerNormGetWorkspaceSize”或者“aclnnLayerNormWithImplModeGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnLayerNorm”或者“aclnnLayerNormWithImplMode”接口执行计算。

- `aclnnStatus aclnnLayerNormGetWorkspaceSize(const aclTensor *input, const aclIntArray *normalizedShape, const aclTensor *weightOptional, const aclTensor *biasOptional, double eps, aclTensor *out, aclTensor *meanOutOptional, aclTensor *rstdOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnLayerNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`
- `aclnnStatus aclnnLayerNormWithImplModeGetWorkspaceSize(const aclTensor *input, const aclIntArray *normalizedShape, const aclTensor *weightOptional, const aclTensor *biasOptional, double eps, aclTensor *out, aclTensor *meanOutOptional, aclTensor *rstdOutOptional, int32_t implMode, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnLayerNormWithImplMode(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- 算子功能：对指定层进行均值为0、标准差为1的归一化计算。
- 计算公式：

  $$
  out = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + eps}} * w + b
  $$

  其中，E[input]表示输入的均值，Var[input]表示输入的方差。

## aclnnLayerNormGetWorkspaceSize

- **参数说明：**

  - input（aclTensor*，计算输入）：公式中的输入`x`，shape为[A1,...,Ai,R1,...,Rj]，其中A1至Ai表示无需norm的维度，R1至Rj表示需norm的维度。支持非连续的Tensor，不支持空Tensor。数据格式支持ND。
     * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT、FLOAT16、BFLOAT16
     * Atlas 训练系列产品、Atlas 200I/500 A2 推理产品：数据类型支持FLOAT、FLOAT16

  - normalizedShape（aclIntArray*，计算输入）：表示需要进行norm计算的维度，数据类型支持INT64，shape为[R1,...,Rj], 长度小于等于输入input的长度，不支持为空。

  - weightOptional（aclTensor*，计算输入）：公式中的输入`w`，可选参数。weightOptional非空时，数据类型与输入input一致或为FLOAT类型，且当biasOptional存在时weightOptional与biasOptional的数据类型相同。shape与normalizedShape相等，为[R1,...,Rj]。支持非连续的Tensor，不支持空Tensor。数据格式支持ND。weightOptional为空时，接口内部会构造一个shape为[R1,...,Rj]，数据全为1的tensor，当biasOptional存在时weightOptional与biasOptional的数据类型相同，biasOptional不存在时weightOptional与输入input的数据类型相同。
     * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT、FLOAT16、BFLOAT16
     * Atlas 训练系列产品、Atlas 200I/500 A2 推理产品：数据类型支持FLOAT、FLOAT16

  - biasOptional（aclTensor*，计算输入）：公式中的输入`b`，可选参数。biasOptional非空时，数据类型与输入input一致或为FLOAT类型，且当weightOptional存在时biasOptional与weightOptional的数据类型相同。shape与normalizedShape相等，为[R1,...,Rj]。支持非连续的Tensor，不支持空Tensor。数据格式支持ND。biasOptional为空时，接口内部会构造一个shape为[R1,...,Rj]，数据全为0的tensor，当weightOptional存在时biasOptional与weightOptional的数据类型相同，weightOptional不存在时biasOptional与输入input的数据类型相同。
     * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT、FLOAT16、BFLOAT16
     * Atlas 训练系列产品、Atlas 200I/500 A2 推理产品：数据类型支持FLOAT、FLOAT16

  - eps（double，计算输入）：公式中的输入`eps`，用于规避除零计算，数据类型支持DOUBLE。

  - out（aclTensor*，计算输出）：公式中的输出`out`。shape需要与input的shape相等，为[A1,...,Ai,R1,...,Rj]。支持非连续的Tensor，不支持空Tensor。数据格式支持ND。
     * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT、FLOAT16、BFLOAT16
     * Atlas 训练系列产品、Atlas 200I/500 A2 推理产品：数据类型支持FLOAT、FLOAT16

  - meanOutOptional（aclTensor*，计算输出）：可选参数。当rstdOutOptional存在时与rstdOutOptional的shape相同，shape为[A1,...,Ai,1,...,1]，Ai后共有j个1，与需要norm的轴长度保持相同。支持非连续的Tensor，不支持空Tensor。数据格式支持ND。
     * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT、FLOAT16、BFLOAT16
     * Atlas 训练系列产品、Atlas 200I/500 A2 推理产品：数据类型支持FLOAT、FLOAT16

  - rstdOutOptional（aclTensor*，计算输出）：可选参数。当meanOutOptional存在时与meanOutOptional的shape相同，shape为[A1,...,Ai,1,...,1]，Ai后共有j个1，与需要norm的轴长度保持相同。支持非连续的Tensor，不支持空Tensor。数据格式支持ND。
     * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT、FLOAT16、BFLOAT16
     * Atlas 训练系列产品、Atlas 200I/500 A2 推理产品：数据类型支持FLOAT、FLOAT16

  - workspaceSize（uint64_t*, 出参）：返回需要在Device侧申请的workspace大小。

  - executor（aclOpExecutor**, 出参）：返回op执行器，包含了算子计算流程。


- **返回值：**

  aclnnStatus：返回状态码。

```
第一段接口完成入参校验，出现以下场景时报错：
返回161001 （ACLNN_ERR_PARAM_NULLPTR）：1. 传入的input、normalizedShape或out为空指针。
返回161002 （ACLNN_ERR_PARAM_INVALID）：1. input、normalizedShape、weightOptional（非空时）、biasOptional（非空时）、out、meanOutOptional（非空时）、rstdOutOptional（非空时），shape的维度超过8维。
                                       2. input、weightOptional（非空时）、biasOptional（非空时）、out、meanOutOptional（非空时）、rstdOutOptional（非空时），数据类型不在支持的范围内。
                                       3. normalizedShape维度小于1维。
                                       4. weightOptional非空且shape与normalizedShape不相等。
                                       5. biasOptional非空且shape与normalizedShape不相等。
                                       6. input的维度小于normalizedShape的维度。
                                       7. input的shape与normalizedShape右对齐时对应维度shape不相等。
```

## aclnnLayerNorm

- **参数说明：**

  - workspace（void*, 入参）：在Device侧申请的workspace内存地址。

  - workspaceSize（uint64_t, 入参）：在Device侧申请的workspace大小，由第一段接口aclnnLayerNormGetWorkspaceSize获取。

  - executor（aclOpExecutor*, 入参）：op执行器，包含了算子计算流程。

  - stream（aclrtStream, 入参）：指定执行任务的AscendCL Stream流。


- **返回值：**

  aclnnStatus：返回状态码。

## aclnnLayerNormWithImplModeGetWorkspaceSize

- **参数说明：**

  - input（aclTensor*，计算输入）：公式中的输入`x`。shape为[A1,...,Ai,R1,...,Rj]，其中A1至Ai表示无需norm的维度，R1至Rj表示需norm的维度。支持非连续的Tensor，不支持空Tensor。数据格式支持ND。
     * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT、FLOAT16、BFLOAT16
     * Atlas 训练系列产品、Atlas 200I/500 A2 推理产品：数据类型支持FLOAT、FLOAT16

  - normalizedShape（aclIntArray*，计算输入）：表示需要进行norm计算的维度，数据类型支持INT64，shape为[R1,...,Rj], 长度小于等于输入input的长度，不支持为空。

  - weightOptional（aclTensor*，计算输入）：公式中的输入`w`，可选参数。数据类型与输入input一致或为FLOAT类型，且当biasOptional存在时与biasOptional的数据类型相同。shape与normalizedShape相等，为[R1,...,Rj]。支持非连续的Tensor，不支持空Tensor。数据格式支持ND。weightOptional为空时，接口内部会构造一个shape为[R1,...,Rj]，数据全为1的tensor，当biasOptional存在时与biasOptional的数据类型相同，biasOptional不存在时与输入input的数据类型相同。
     * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT、FLOAT16、BFLOAT16
     * Atlas 训练系列产品、Atlas 200I/500 A2 推理产品：数据类型支持FLOAT、FLOAT16

  - biasOptional（aclTensor*，计算输入）：公式中的输入`b`，可选参数。数据类型与输入input一致或为FLOAT类型，且当weightOptional存在时与weightOptional的数据类型相同。shape与normalizedShape相等，为[R1,...,Rj]。支持非连续的Tensor，不支持空Tensor。数据格式支持ND。biasOptional为空时，接口内部会构造一个shape为[R1,...,Rj]，数据全为0的tensor，当weightOptional存在时与weightOptional的数据类型相同，weightOptional不存在时与输入input的数据类型相同。
     * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT、FLOAT16、BFLOAT16
     * Atlas 训练系列产品、Atlas 200I/500 A2 推理产品：数据类型支持FLOAT、FLOAT16

  - eps（double，计算输入）：公式中的输入`eps`，用于规避除零计算，数据类型支持DOUBLE。

  - out（aclTensor*，计算输出）：公式中的输出`out`。shape需要与input的shape相等，为[A1,...,Ai,R1,...,Rj]。支持非连续的Tensor，不支持空Tensor。数据格式支持ND。
     * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT、FLOAT16、BFLOAT16
     * Atlas 训练系列产品、Atlas 200I/500 A2 推理产品：数据类型支持FLOAT、FLOAT16

  - meanOutOptional（aclTensor*，计算输出）：可选参数。当rstdOutOptional存在时与rstdOutOptional的shape相同，shape为[A1,...,Ai,1,...,1]，Ai后共有j个1，与需要norm的轴长度保持相同。支持非连续的Tensor，不支持空Tensor。数据格式支持ND。
     * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT、FLOAT16、BFLOAT16
     * Atlas 训练系列产品、Atlas 200I/500 A2 推理产品：数据类型支持FLOAT、FLOAT16

  - rstdOutOptional（aclTensor*，计算输出）：可选参数。当meanOutOptional存在时与meanOutOptional的shape相同，shape为[A1,...,Ai,1,...,1]，Ai后共有j个1，与需要norm的轴长度保持相同。支持非连续的Tensor，不支持空Tensor。数据格式支持ND。
     * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT、FLOAT16、BFLOAT16
     * Atlas 训练系列产品、Atlas 200I/500 A2 推理产品：数据类型支持FLOAT、FLOAT16

  - implMode（int32_t, 计算输入）：精度模式，用于指定kernel选择对应的计算模式，默认实现为高精度模式(0), 高性能模式分为高性能模式(1)/保持FLOAT16计算模式(2)，高性能模式谨慎使用，有精度损失，保持FLOAT16计算模式仅支持所有输入同时为FLOAT16，且计算精度最低。

  - workspaceSize（uint64_t*, 出参）：返回需要在Device侧申请的workspace大小。

  - executor（aclOpExecutor**, 出参）：返回op执行器，包含了算子计算流程。


- **返回值：**

  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的input、normalizedShape或out为空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. input、normalizedShape、weightOptional非空、biasOptional非空、out、meanOutOptional非空或rstdOutOptional非空时的shape超过8维。
                                        2. input、weightOptional非空、biasOptional非空、out、meanOut非空或rstdOut非空时的数据类型不在支持的范围内。
                                        3. normalizedShape维度小于1维。
                                        4. weightOptional非空且shape与normalizedShape不相等。
                                        5. biasOptional非空且shape与normalizedShape不相等。
                                        6. input的维度小于normalizedShape的维度。
                                        7. input的shape与normalizedShape右对齐时对应维度shape不相等。
                                        8. implMode的取值不在0，1和2范围内。
                                        9. implMode的取值为2且输入的数据类型不全部为FLOAT16。
  ```

## aclnnLayerNormWithImplMode

- **参数说明：**

  - workspace（void*, 入参）：在Device侧申请的workspace内存地址。

  - workspaceSize（uint64_t, 入参）：在Device侧申请的workspace大小，由第一段接口aclnnLayerNormWithImplModeGetWorkspaceSize获取。

  - executor（aclOpExecutor*, 入参）：op执行器，包含了算子计算流程。

  - stream（aclrtStream, 入参）：指定执行任务的AscendCL Stream流。


- **返回值：**

  aclnnStatus：返回状态码。

## 约束与限制

无。
