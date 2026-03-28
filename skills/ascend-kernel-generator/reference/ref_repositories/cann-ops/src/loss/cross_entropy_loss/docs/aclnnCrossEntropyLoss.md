# aclnnCrossEntropyLoss
## 支持的产品型号
- Atlas A2 训练系列产品/Atlas 800I A2推理产品
- Atlas A3 训练系列产品/Atlas 800I A3推理产品

## 接口原型

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnCrossEntropyLossGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnCrossEntropyLoss”接口执行计算。

- `aclnnStatus aclnnCrossEntropyLossGetWorkspaceSize(const aclTensor* input, const aclTensor* target, const aclTensor* weightOptional, char* reductionOptional, int64_t ignoreIndex, double labelSmoothing, double lseSquareScaleForZloss, bool returnZloss, const aclTensor *lossOut, const aclTensor *logProbOut, const aclTensor *zlossOut, const aclTensor *lseForZlossOut, uint64_t *workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnCrossEntropyLoss(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- 算子功能：计算输入的交叉熵损失。
- 计算表达式：
  
  reduction = mean时，交叉熵损失loss的计算公式为：
  $$
  l_n = -weight_{y_n}*log\frac{exp(x_{n,y_n})}{\sum_{c=1}^Cexp(x_{n,c})}*1\{y_n\ !=\ ignoreIndex \}
  $$
  $$
  loss=\begin{cases}\sum_{n=1}^N\frac{1}{\sum_{n=1}^Nweight_{y_n}*1\{y_n\ !=\ ignoreIndex \}}l_n,&\text{if reduction = ‘mean’} \\\sum_{n=1}^Nl_n,&\text {if reduction = ‘sum’ }\\\{l_0,l_1,...,l_n\},&\text{if reduction = ‘None’ }\end{cases}
  $$
  log\_prob计算公式为：
  $$
  lse_n = log*\sum_{c=1}^{C}exp(x_{n,c})
  $$
  $$
  logProb_{n,c} = x_{n,c} - lse_n
  $$
  zloss计算公式为：
  $$
  zloss_n = lseSquareScaleForZloss * （lse_n）^2 
  $$
  其中，N为batch数，C为标签数。
  
## aclnnCrossEntropyLossGetWorkspaceSize

- **参数说明：**

  - input(aclTensor*, 计算输入)：表示输入，公式中的`input`，Device侧的aclTensor。数据类型支持FLOAT、FLOAT16、BFLOAT16。shape为($N, C$)，$N$为批处理大小，$C$为标签数，必须大于0。[数据格式](common/数据格式.md)支持ND。
  - target(aclTensor*, 计算输入)：表示标签，公式中的`y`，Device侧的aclTensor。数据类型支持INT64。shape为($N$)，N与input第零维相等，数值在[0, C)之间。[数据格式](common/数据格式.md)支持ND。
  - weightOptional(aclTensor*, 计算输入)：表示为每个类别指定的缩放权重，公式中的`weight`。为inputLengths中的元素，Device侧的aclTensor。数据类型支持FLOAT。shape为（$C$）。如果不给定，则不对target加权。[数据格式](common/数据格式.md)支持ND。
  - reduction(char*, 计算输入)：表示loss的归约方式。Host侧的String，支持["mean", "sum", "none"]。
  - ignoreIndex(int, 计算输入)：指定忽略的标签。Host侧的整型。数值必须小于$C$，当小于零时视为无忽略标签。
  - labelSmoothing(double, 计算输入)：表示计算loss时的平滑量。Host侧的浮点型。数值在[0.0, 1.0)之间。
  - lseSquareScaleForZloss(double, 计算输入)：表示zloss计算所需的scale。Host侧的浮点型。公式中的`lse_square_scale_for_zloss`。数值在[0, 1)之间。当前仅支持传入nulltpr。
  - returnZloss(bool, 计算输入)：控制是否返回zloss输出。Host侧的布尔值。需要输出zLoss时传入True，否则传入False。当前仅支持传入nulltpr。
  - lossOut(aclTensor*，计算输出)：表示输出损失。Device侧的aclTensor。数据类型与input相同。reduction为"None"时，shape为[N]，与input第零维一致；否则shape为[1]。[数据格式](common/数据格式.md)支持ND。
  - logProbOut(aclTensor*，计算输出)：输出给反向计算的输出。Device侧的aclTensor。数据类型与input相同。shape为[$N,C$]，与input一致。[数据格式](common/数据格式.md)支持ND。
  - zlossOut(aclTensor*，计算输出)：表示辅助损失。Device侧的aclTensor。数据类型与input相同。shape为与loss一致。[数据格式](common/数据格式.md)支持ND。当return_zloss为True时，输出zloss，否则输出为None。当前暂不支持。
  - lseForZlossOut(aclTensor*，计算输出)：表示zloss场景输出给反向的Tensor，lseSquareScaleForZloss为0时输出为None。Device侧的aclTensor。数据类型与input相同。shape为[N]，与input的第零维一致。[数据格式](common/数据格式.md)支持ND。当前暂不支持。
  - workspaceSize(uint64_t*, 出参)：返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor**, 出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

    ```
    第一段接口完成入参校验，出现以下场景时报错：
    161001 (ACLNN_ERR_PARAM_NULLPTR): 1. 传入的input、target、loss、logProb、zloss、lseForZloss是空指针。
    ```

## aclnnCrossEntropyLoss

- **参数说明：**

  - workspace(void*, 入参): 在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t, 入参): 在Device侧申请的workspace大小，由第一段接口aclnnCrossEntropyLossGetWorkspaceSize获取。
  - executor(aclOpExecutor*, 入参): op执行器，包含了算子计算流程。
  - stream(aclrtStream, 入参): 指定执行任务的AscendCL Stream流。

- **返回值：**

  - aclnnStatus: 返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束与限制
  - target仅支持类标签索引，不支持概率输入。
  - 当前暂不支持zloss相关功能。lseSquareScaleForZloss、returnZloss仅支持传入nullptr。
  - input第零维N需满足N<200000。
