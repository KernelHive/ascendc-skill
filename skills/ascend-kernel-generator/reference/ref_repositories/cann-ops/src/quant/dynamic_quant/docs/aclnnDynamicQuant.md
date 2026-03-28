# aclnnDynamicQuant

## 支持的产品型号

- Atlas 推理系列产品。
- Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件。

## 接口原型
每个算子分为两段式接口，必须先调用“aclnnDynamicQuantGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnDynamicQuant”接口执行计算。

* `aclnnstatus aclnnDynamicQuantGetWorkspaceSize(const aclTensor* x, const aclTensor* smoothScalesOptional, const aclTensor* yOut, const aclTensor* scaleOut, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnstatus aclnnDynamicQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- 算子功能：为输入张量进行per-token对称动态量化。

- 计算公式：
  - 若不输入smoothScalesOptional，则

  $$
   scaleOut=row\_max(abs(x))/127
  $$

  $$
   yOut=round(x/scalOut)
  $$
  - 若输入smoothScalesOptional，则
  $$
    input = x\cdot smoothScalesOptional
  $$
  $$
   scaleOut=row\_max(abs(input))/127
  $$

  $$
   yOut=round(input/scalOut)
  $$
  其中row\_max代表每行求最大值。


## aclnnDynamicQuantGetWorkspaceSize

- **参数说明：**

  - x（aclTensor*, 计算输入）：算子输入的Tensor，shape维度要大于1，Device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16，支持非连续的Tensor，数据格式支持ND。
  - smoothScalesOptional（aclTensor*, 计算输入）：算子输入的smoothScales，shape维度是x的最后一维，Device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16，并且数据类型要和x保持一致，支持非连续的Tensor，数据格式支持ND。
  - yOut（aclTensor*, 计算输出）：量化后的输出Tensor，shape维度和x保持一致，Device侧的aclTensor，数据类型支持INT8，暂不支持非连续的Tensor，数据格式支持ND。
  - scaleOut（aclTensor*, 计算输出）：量化使用的scale，其shape为x的shape剔除最后一维，Device侧的aclTensor，数据类型支持FLOAT，暂不支持非连续的Tensor，数据格式支持ND。
  - workspaceSize（uint64_t\*, 出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*, 出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus: 返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR)：传入的x或out参数是空指针。
  返回161002 (ACLNN_ERR_PARAM_INVALID)：参数的数据类型、数据格式、维度等不在不在支持范围内。
  返回561001 (ACLNN_ERR_INNER_CREATE_EXECUTOR)：内部创建aclOpExecutor失败。
  ```

## aclnnDynamicQuant

- **参数说明：**
  - workspace(void \*, 入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnDynamicQuantGetWorkspaceSize获取。
  - executor(aclOpExecutor \*, 入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream, 入参)：指定执行任务的AscendCL Stream流。

- **返回值：**

  aclnnStatus: 返回状态码。

## 约束与限制
无

## 调用示例

详见[DynamicQuant自定义算子样例说明算子调用章节](../README.md#算子调用)
