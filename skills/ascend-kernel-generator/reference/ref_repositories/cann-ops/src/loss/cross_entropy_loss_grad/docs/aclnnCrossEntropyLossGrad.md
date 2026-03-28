# aclnnCrossEntropyLossGrad
## 支持的产品型号

- Atlas A2 训练系列产品/Atlas 800I A2推理产品
- Atlas A3 训练系列产品/Atlas 800I A3推理产品

## 接口原型

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnCrossEntropyLossGradGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnCrossEntropyLossGrad”接口执行计算。

* `aclnnStatus aclnnCrossEntropyLossGradGetWorkspaceSize(const aclTensor *gradLoss, const aclTensor *logProb, const aclTensor *target, const aclTensor *weightOptional, const aclTensor *gradZlossOptional, const aclTensor *lseForZlossOptional, char *reductionOptional, int64_t ignoreIndex, double labelSmoothing, double lseSquareScaleForZloss, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnCrossEntropyLossGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- **算子功能**：aclnnCrossEntropyLoss的反向传播。
- **计算公式**：

$$
ignoreMask_{target(t)}=\begin{cases}
1, &target(t) ≠ ignoreIndex \\
0, &target(t) = ignoreIndex
\end{cases}
$$

$$
smoothLossGrad=\begin{cases}
grad / sum(weight_{target}* ignoreMask) * labelSmoothing / C, &redutcion = mean \\
grad * labelSmoothing / C, &redutcion = sum \\
grad * labelSmoothing / C, &redutcion = none
\end{cases}
$$

$$
lossOutGrad=\begin{cases}
grad * (1-labelSmoothing) / sum(weight_{target}* ignoreMask) * ignoreMask, &redutcion = mean \\
grad * (1-labelSmoothing) * ignoreMask, &redutcion = sum \\
grad * (1-labelSmoothing) * ignoreMask, &redutcion = none
\end{cases}
$$

$$
nllLossGrad = lossOutGrad * weight_{target}
$$

$$
logSoftmaxGradLossOutSubPart = exp(logProb) * nllLossGrad
$$

$$
predictionsGradLossOut_{ij}=\begin{cases}
nllLossGrad_i, & j=target(i)  \\
0, & j ≠ target(i) 
\end{cases}
$$

$$
predictionsGradLossOut = logSoftmaxGradLossOutSubPart - predictionsGradLossOut
$$

$$
smoothLossGrad = smoothLossGrad * ignoreMask
$$

$$
logSoftmaxGradSmoothLoss = smoothLossGrad * weight
$$

$$
predictionsGradSmoothLoss = exp(logProb) * sum(logSoftmaxGradSmoothLoss) - logSoftmaxGradSmoothLoss
$$

不涉及zloss场景输出：

$$
xGrad_{out} = predictionsGradLossOut + predictionsGradSmoothLoss
$$

zloss场景：

$$
gradZ=\begin{cases}
grad + gradZloss, & 传入gradZloss  \\
grad, & 不传gradZloss
\end{cases}
$$

$$
zlossGrad=\begin{cases}
gradZ / sum(ignoreMask), & &redutcion = mean  \\
gradZ, & &redutcion = sum \\
gradZ, & &redutcion = none
\end{cases}
$$

$$
lseGrad = 2 * lseSquareScaleForZloss * lseForZloss * ignoreMask * zlossGrad
$$

$$
zlossOutputGrad = exp(logProb) * lseGrad
$$

zloss场景输出：

$$
xGrad_{out} = xGrad_{out} + zlossOutputGrad
$$

## aclnnCrossEntropyLossGradGetWorkspaceSize

- **参数说明：**
  
  - gradLoss（aclTensor\*，计算输入）：Device侧的aclTensor，正向输出loss的梯度。参数与公式中grad对应。当reduction为none时，要求为一个维度为1D的Tensor，shape为 (N,)，$N$为批处理大小；当reduction为mean/sum时，要求为一个维度为0D的Tensor。数据类型支持FLOAT16、FLOAT、BFLOAT16，[数据格式](common/数据格式.md)要求为ND。
  - logProb（aclTensor\*，计算输入）：Device侧的aclTensor，正向输入的logSoftmax计算结果，要求为一个维度为2D的Tensor，shape为 (N, C)，$C$为标签数，必须大于0。数据类型支持FLOAT16、FLOAT、BFLOAT16，[数据格式](common/数据格式.md)要求为ND。
  - target（aclTensor\*，计算输入）：Device侧的aclTensor，类索引，要求为一个维度为1D的Tensor，shape为 (N,)，取值范围为[0, C)。数据类型支持INT64，[数据格式](common/数据格式.md)要求为ND。
  - weightOptional（aclTensor\*，计算输入）：Device侧的aclTensor，可选输入，要求shape为一个1D的Tensor，shape为(C,)。数据类型支持FLOAT32，[数据格式](common/数据格式.md)要求为ND。
  - gradZlossOptional（aclTensor\*，计算输入）：Device侧的aclTensor，可选输入，当前仅支持传入nullptr。参数与公式中gradZloss对应。zloss相关输入，如果正向有zloss的额外输出，反向有个grad_zloss的输入。当reduction为none时，要求为一个维度为1D的Tensor，shape为 (N,)；当reduction为mean/sum时，要求为一个维度为0D的Tensor。数据类型支持FLOAT16、FLOAT、BFLOAT16，[数据格式](common/数据格式.md)要求为ND。
  - lseForZlossOptional（aclTensor\*，计算输入）：Device侧的aclTensor，可选输入。zloss相关输入，如果lse_square_scale_for_zloss非0，正向额外输出的lse_for_zloss中间结果给反向用于计算lse。要求为一个维度为1D的Tensor，shape为 (N,)。当前只支持传入nullptr。数据类型支持FLOAT16、FLOAT、BFLOAT16，[数据格式](common/数据格式.md)要求为ND。
  - reduction（char* , 计算输入）：指定要应用于输出的缩减。Host侧的字符串。'none'：不应用缩减，'mean'：取输出的加权平均值，'sum'：求和输出。
  - ignoreIndex（int64_t, 计算输入）：指定忽略不影响输入梯度的目标值。Host侧的整型。数值必须小于**C**，当小于零时视为无忽略标签。
  - labelSmoothing（double, 计算输入）：表示计算损失时的平滑量。Host侧的浮点型。取值范围在[0.0, 1.0]的浮点数，其中0.0表示不平滑。当前仅支持输入0.0。
  - lseSquareScaleForZloss（double, 计算输入）：zloss相关属性，0.0走pytorch原生分支，非0.0走zloss新分支。当前仅支持输入0.0。
  - out（aclTensor\*，计算输出）：梯度计算结果，要求是一个2D的Tensor，shape为（N, C）。数据类型同gradLoss，支持BFLOAT16、FLOAT16、FLOAT32，[数据格式](common/数据格式.md)要求为ND。
  - workspaceSize（uint64\_t\*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

- **返回值**：
  
	aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

	```
	第一段接口完成入参校验，出现以下场景时报错：
	返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的gradLoss、logProb、target、xGradOut为空指针。
	返回161002（ACLNN_ERR_PARAM_INVALID）：1. gradLoss、logProb、target、weightOptional、gradZlossOptional、lseForZlossOptional的数据类型不在支持的范围内。
	```

## aclnnCrossEntropyLossGrad

- **参数说明：**
  
  - workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnCrossEntropyLossGradGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream,入参）：指定执行任务的AscendCL stream流。

- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](./common/aclnn返回码.md)。

## 约束与限制

  - target仅支持类标签索引，不支持概率输入。
  - gradLoss、logProb、gradZlossOptional、lseForZlossOptional、xGradOut数据类型需保持一致。
  - 当前暂不支持zloss功能。gradZlossOptional、lseForZlossOptional不支持传入，且lseSquareScaleForZloss仅支持输入0.0。
  - logProb第零维N需满足N<200000。
