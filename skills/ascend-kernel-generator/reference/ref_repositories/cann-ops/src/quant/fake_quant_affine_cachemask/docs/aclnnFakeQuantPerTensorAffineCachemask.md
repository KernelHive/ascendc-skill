# aclnnFakeQuantPerTensorAffineCachemask

## 支持的产品型号

- Atlas A2 训练系列产品/Atlas 800I A2 推理产品

## 接口原型

每个算子分为两段式接口，必须先调用“aclnnFakeQuantPerTensorAffineCachemaskGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnFakeQuantPerTensorAffineCachemask”接口执行计算。

* `aclnnStatus aclnnFakeQuantPerTensorAffineCachemaskGetWorkspaceSize(const aclTensor* self, const aclTensor* scale, const aclTensor* zeroPoint, float fakeQuantEnbled, int64_t quantMin, int64_t quantMax, aclTensor* out, aclTensor* mask, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnStatus aclnnFakeQuantPerTensorAffineCachemask(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## 功能描述

- 算子功能：
  - fake_quant_enabled >= 1: 对于输入数据self，使用scale和zero_point对输入self进行伪量化处理，并根据quant_min和quant_max对伪量化输出进行值域更新，最终返回结果out及对应位置掩码mask。
  - fake_quant_enabled < 1: 返回结果out为self.clone()对象，掩码mask为全True。
- 计算公式：在fake_quant_enabled >= 1的情况下，根据算子功能先计算临时变量qval，再计算得出out和mask。

  $$
  qval = Round(std::nearby\_int(self / scale) + zero\_point)
  $$

  $$
  out = (Min(quant\_max, Max(quant\_min, qval)) - zero\_point) * scale
  $$

  $$
  mask = (qval >= quant\_min)   \&  (qval <= quant\_max)
  $$

## aclnnFakeQuantPerTensorAffineCachemaskGetWorkspaceSize

- **参数说明：**
  - self(aclTensor*, 计算输入)：Device侧的aclTensor，数据类型支持FLOAT16、FLOAT32。支持非连续的Tensor，数据格式支持ND。
  - scale(aclTensor*, 计算输入)：Device侧的aclTensor，表示输入伪量化的缩放系数。数据类型支持FLOAT16、FLOAT32，size大小为1。
  - zeroPoint(aclTensor*, 计算输入)：Device侧的aclTensor，表示输入伪量化的零基准参数。数据类型支持INT32，size大小为1。
  - fakeQuantEnbled(float, 计算输入)：Host侧的浮点型，表示是否进行伪量化计算。
  - quantMin(int64_t, 计算输入)：Host侧的整型，表示输入数据伪量化后的最小值。
  - quantMax(int64_t, 计算输入)：Host侧的整型，表示输入数据伪量化后的最大值。
  - out(aclTensor\*, 计算输出)：Device侧的aclTensor，数据类型支持FLOAT16、FLOAT32，支持非连续Tensor，数据格式支持ND。
  - mask(aclTensor\*, 计算输出)：Device侧的aclTensor，数据类型支持BOOL，支持非连续Tensor，数据格式支持ND。
  - workspaceSize(uint64_t\*, 出参)：返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor\*\*, 出参)：返回op执行器，包含了算子计算流程。
- **返回值：**
  aclnnStatus： 返回状态码。
  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR): 1. 传入的self、scale、zeroPoint、out或mask是空指针。
  返回161002 (ACLNN_ERR_PARAM_INVALID): 1. self、scale、zeroPoint、out或mask的数据类型不在支持的范围之内。
                                        2. scale或zeroPoint的size大小不是1。
                                        3. quantMin大于quantMax。
  ```

## aclnnFakeQuantPerTensorAffineCachemask

- **参数说明：**
  - workspace(void\*, 入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnFakeQuantPerChannelAffineCachemaskGetWorkspaceSize获取。
  - executor(aclOpExecutor\*, 入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream, 入参)：指定执行任务的AscendCL Stream流。
- **返回值：**
  aclnnStatus：返回状态码。

## 约束与限制
无
