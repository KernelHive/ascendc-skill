# aclnnGroupQuant

## 支持的产品型号

- Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件。
- Atlas A3 训练系列产品/Atlas A3 推理系列产品。

## 接口原型
每个算子分为两段式接口，必须先调用“aclnnGroupQuantGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGroupQuant”接口执行计算。

- `aclnnStatus aclnnGroupQuantGetWorkspaceSize(const aclTensor* x, const aclTensor* scale, const aclTensor* groupIndex, const aclTensor* offsetOptional, int32_t dstType, aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnGroupQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- 算子功能：对输入x进行分组量化操作。
- 计算公式：

  $$
  y = round((x * scale) + offsetOptional)
  $$

## aclnnGroupQuantGetWorkspaceSize

- **参数说明：**

  - x（aclTensor*，计算输入）：公式中的`x`，Device侧的aclTensor，需要做量化的输入。shape为2维，支持非连续的Tensor，不支持空tensor。数据格式支持ND。数据类型支持FLOAT32，FLOAT16，BFLOAT16。如果`dstType`为3(INT32)，Shape的最后一维需要能被8整除；如果`dstType`为29(INT4)，Shape的最后一维需要能被2整除。
  - scale（aclTensor*，计算输入）：公式中的`scale`，Device侧的aclTensor，量化中的scale值。数据类型支持FLOAT32，FLOAT16，BFLOAT16。`scale`为2维张量(`scale`的第1维与x的第1维相等)。支持非连续的Tensor，不支持空tensor。数据格式支持ND。
  - groupIndex（aclTensor*，计算输入）：Device侧的aclTensor，Group量化中的groupIndex值。数据类型支持INT32，INT64。`groupIndex`为1维张量(维度与scale的第0维相等)。支持非连续的Tensor，不支持空tensor。数据格式支持ND。
  - offsetOptional（aclTensor*，计算输入）：公式中的`offsetOptional`，可选参数，Device侧的aclTensor，量化中的offset值。数据类型支持FLOAT32，FLOAT16，BFLOAT16，并且数据类型与`scale`一致。`offsetOptional`为1个数。
  - dstType（int32_t，计算输入）：Host侧整型属性，指定输出的数据类型，该属性数据类型支持：INT32。支持取值2，3，29，分别表示INT8，INT32，INT4。
  - y（aclTensor*，计算输出）: 公式中的`y`，Device侧的aclTensor。shape为2维，数据类型支持INT8，INT32，INT4。类型为INT32时，Shape的最后一维是`x`最后一维的1/8，其余维度和x一致; 其他类型时，Shape与`x`一致。支持非连续的Tensor，不支持空tensor。数据格式支持ND。
  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的x、scale、groupIndex、y是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. x、scale、groupIndex、offsetOptional、y的数据类型或数据格式不在支持的范围之内。
                                        2. scale与offsetOptional的数据类型不一致。
                                        3. x、scale、groupIndex、offsetOptional、y的shape不满足限制要求。
                                        4. y的数据类型为INT4时，x的shape尾轴大小不是偶数。
                                        5. y的数据类型为INT32时，x的shape尾轴不是y的shape尾轴大小的8倍，或者x与y的shape的非尾轴的大小不一致。
  ```

## aclnnGroupQuant

- **参数说明：**

  - workspace（void*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnGroupQuantGetWorkspaceSize获取。
  - executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。

- **返回值：**

  aclnnStatus：返回状态码。

## 约束与限制
- 输入`scale`与输入`offsetOptional`的数据类型一致。
- 如果属性`dstType`为29(INT4)，那么输入`x`的Shape的最后一维需要能被2整除。
- 如果属性`dstType`为3(INT32)，那么输入`x`的Shape的最后一维需要能被8整除，输入`x`的shape尾轴是输出`y`的shape尾轴大小的8倍。
- 输入`groupIndex`必须是非递减序列，最小值不能小于0，最大值必须与输入`x`的shape的第0维大小相等；当不满足约束限制时，参数`groupIndex`不校验。
- 输入`scale`的第0维大小不支持为0。
- 输入`offsetOptional`的shape当前仅支持[1, ]或[ , ]。

