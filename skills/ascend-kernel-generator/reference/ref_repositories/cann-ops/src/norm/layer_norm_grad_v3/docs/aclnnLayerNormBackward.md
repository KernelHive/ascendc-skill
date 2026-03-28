# aclnnLayerNormBackward

## 支持的产品型号
- Atlas 推理系列产品。
- Atlas 训练系列产品。
- Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件。
- Atlas A3 训练系列产品/Atlas A3 推理系列产品。

## 接口原型

每个算子分为两段式接口，必须先调用“aclnnLayerNormBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnLayerNormBackward”接口执行计算。

- `aclnnStatus aclnnLayerNormBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *input, const aclIntArray *normalizedShape, const aclTensor *mean, const aclTensor *rstd, const aclTensor *weightOptional, const aclTensor *biasOptional, const aclBoolArray *outputMask, aclTensor *gradInputOut, aclTensor *gradWeightOut, aclTensor *gradBiasOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnLayerNormBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

算子功能：LayerNorm的反向传播。

## aclnnLayerNormBackwardGetWorkspaceSize

- **参数说明：**

  - gradOut（aclTensor*，计算输入）：反向计算的梯度tensor，与输入input的数据类型相同。shape与input的shape相等，为[A1,...,Ai,R1,...,Rj]，shape长度大于等于normalizedShape的长度。支持非连续的Tensor，不支持空Tensor。数据格式支持ND。
     * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：FLOAT、FLOAT16、BFLOAT16
     * Atlas 推理系列产品、Atlas 训练系列产品：FLOAT、FLOAT16

  - input（aclTensor*，计算输入）：正向计算的首个输入，与输入gradOut的数据类型相同。shape与gradOut的shape相等，为[A1,...,Ai,R1,...,Rj]，shape长度大于等于normalizedShape的长度。支持非连续的Tensor，不支持空Tensor。数据格式支持ND。
     * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：FLOAT、FLOAT16、BFLOAT16
     * Atlas 推理系列产品、Atlas 训练系列产品：FLOAT、FLOAT16

  - normalizedShape（aclIntArray*，计算输入）：表示需要进行norm计算的维度，数据类型支持INT64，shape为[R1,...,Rj]，长度小于等于输入input的长度，不支持为空。

  - mean（aclTensor*，计算输入）：正向计算的第二个输出，表示input的均值，与输入rstd的数据类型相同且位宽不低于输入input的数据类型位宽。shape与rstd的shape相等，为[A1,...,Ai,1,...,1]，Ai后共有j个1，与需要norm的轴长度保持相同。支持非连续的Tensor，不支持空Tensor。数据格式支持ND。
     * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：FLOAT、FLOAT16、BFLOAT16
     * Atlas 推理系列产品、Atlas 训练系列产品：FLOAT、FLOAT16

  - rstd（aclTensor*，计算输入）：正向计算的第三个输出，表示input的标准差的倒数，与输入mean的数据类型相同且位宽不低于输入input的数据类型位宽。shape与mean的shape相等，为[A1,...,Ai,1,...,1]，Ai后共有j个1，与需要norm的轴长度保持相同。支持非连续的Tensor，不支持空Tensor。数据格式支持ND。
     * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：FLOAT、FLOAT16、BFLOAT16
     * Atlas 推理系列产品、Atlas 训练系列产品：FLOAT、FLOAT16

  - weightOptional（aclTensor*，计算输入）：权重tensor，可选参数。weightOptional非空时，数据类型与输入input一致或为FLOAT类型，且当biasOptional存在时与biasOptional的数据类型相同。shape与normalizedShape相等，为[R1,...,Rj]。支持非连续的Tensor，不支持空Tensor。数据格式支持ND。weightOptional为空时，需要构造一个shape为[R1,...,Rj]，数据类型与输入input相同，数据全为1的tensor。
     * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：FLOAT、FLOAT16、BFLOAT16
     * Atlas 推理系列产品、Atlas 训练系列产品：FLOAT、FLOAT16

  - biasOptional（aclTensor*，计算输入）：偏置tensor，可选参数。biasOptional非空时，数据类型与输入input一致或为FLOAT类型，且当weightOptional存在时与weightOptional的数据类型相同。shape与normalizedShape相等，为[R1,...,Rj]。支持非连续的Tensor，不支持空Tensor。数据格式支持ND。biasOptional为空时，不做任何处理。
     * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：FLOAT、FLOAT16、BFLOAT16
     * Atlas 推理系列产品、Atlas 训练系列产品：FLOAT、FLOAT16

  - outputMask（aclBoolArray*，计算输入）：数据类型支持BOOL，长度固定为3，取值为True时表示对应位置的输出非空。

  - gradInputOut（aclTensor*，可选输出）：由outputMask的第0个元素控制是否输出，outputMask第0个元素为True时会进行输出，与输入input的数据类型相同。shape与input的shape相等，为[A1,...,Ai,R1,...,Rj]。支持非连续的Tensor，不支持空Tensor。数据格式支持ND。
     * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：FLOAT、FLOAT16、BFLOAT16
     * Atlas 推理系列产品、Atlas 训练系列产品：FLOAT、FLOAT16

  - gradWeightOut（aclTensor*，可选输出）：由outputMask的第1个元素控制是否输出，outputMask第1个元素为True时会进行输出，与输入weightOptional的数据类型相同。shape与gradBiasOut的shape相等，为[R1,...,Rj]。支持非连续的Tensor，不支持空Tensor。数据格式支持ND。
     * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：FLOAT、FLOAT16、BFLOAT16
     * Atlas 推理系列产品、Atlas 训练系列产品：FLOAT、FLOAT16

  - gradBiasOut（aclTensor*，可选输出）：由outputMask的第2个元素控制是否输出，outputMask第2个元素为True时会进行输出，与输入weightOptional的数据类型相同。shape与gradWeightOut的shape相等，为[R1,...,Rj]。支持非连续的Tensor，不支持空Tensor。数据格式支持ND。
     * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：FLOAT、FLOAT16、BFLOAT16
     * Atlas 推理系列产品、Atlas 训练系列产品：FLOAT、FLOAT16

  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。

  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。


- **返回值：**

  aclnnStatus：返回状态码。

```
第一段接口完成入参校验，出现以下场景时报错：
返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的gradOut、input、normalizedShape、mean、rstd、outputMask为空指针。
                                      2. outputMask[0]为True且gradInputOut为空指针。
                                      3. outputMask[1]为True且gradWeightOut为空指针。
                                      4. outputMask[2]为True且gradBiasOut为空指针。
返回161002（ACLNN_ERR_PARAM_INVALID）：1. gradOut、input、mean、rstd、weightOptional非空或biasOptional非空时的数据类型不在支持范围内。
                                      2. gradOut的shape与input的shape不相等。
                                      3. normalizedShape维度小于1维。
                                      4. mean的shape乘积与input从第0根轴到第len(input) - len(normalizedShape)轴的乘积不相等。
                                      5. rstd的shape乘积与input从第0根轴到第len(input) - len(normalizedShape)轴的乘积不相等。
                                      6. weightOptional非空且shape与normalizedShape不相等。
                                      7. biasOptional非空且shape与normalizedShape不相等。
                                      8. input的维度小于normalizedShape的维度。
                                      9. input的shape与normalizedShape右对齐时对应维度shape不相等。
                                      10.outputMask的长度不为3。
```

## aclnnLayerNormBackward

- **参数说明：**

  - workspace（void*，入参）：在Device侧申请的workspace内存地址。

  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnLayerNormBackwardGetWorkspaceSize获取。

  - executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。

  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。


- **返回值：**

  aclnnStatus：返回状态码。

## 约束与限制

无。
