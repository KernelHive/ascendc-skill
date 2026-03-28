# aclnnLinspace

## 支持的产品型号

- Atlas 推理系列产品/Atlas 训练系列产品/Atlas A2 训练系列产品/Atlas A3 训练系列产品/Atlas 800I A2 推理产品。

## 接口原型

每个算子分为两段式接口，必须先调用“aclnnLinspaceGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnLinspace”接口执行计算。

- `aclnnStatus aclnnLinspaceGetWorkspaceSize(const aclScalar *start, const aclScalar *end, int64_t steps, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnLinspace(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- 算子功能：创建一个大小为steps的1维向量，其值从start起始到end结束（包含）线性均匀分布。

- 计算公式：


$$
out = (start, start + \frac{end - start}{steps - 1},...,start + (steps - 2) * \frac{end - start}{steps -1}, end)
$$
## aclnnLinspaceGetWorkspaceSize

- **参数说明：**

  * start(aclScalar *，计算输入)：获取值的范围的起始位置，数据类型支持FLOAT，数据格式支持ND。

  * end(aclScalar *，计算输入)：获取值的范围的结束位置，数据类型支持FLOAT，数据格式支持ND。

  * steps(int64_t *，计算输入)：获取值的步长，数据类型支持INT64，数据格式支持ND。需要满足steps大于等于0。

  * out(aclTensor *，计算输出)：指定的输出Tensor，数据类型支持FLOAT，数据格式支持ND。

  * workspaceSize(uint64_t*, 出参)：返回需要在Device侧申请的workspace大小。

  * executor(aclOpExecutor**, 出参)：返回op执行器，包含了算子计算流程。


- **返回值：**

  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR)：1. 传入的start、end、steps或out是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. start、end、steps或out的数据类型不在支持的范围之内。
                                        2. steps小于0。
  ```

## aclnnLinspace

- **参数说明：**

  * workspace(void*, 入参)：在Device侧申请的workspace内存地址。

  * workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnLinspaceGetWorkspaceSize获取。

  * executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。

  * stream(aclrtStream, 入参)：指定执行任务的AscendCL Stream流。


- **返回值：**

  aclnnStatus：返回状态码。

## 约束与限制

无。

## 调用示例

详见[LinSpace自定义算子样例说明算子调用章节](../README.md#算子调用)