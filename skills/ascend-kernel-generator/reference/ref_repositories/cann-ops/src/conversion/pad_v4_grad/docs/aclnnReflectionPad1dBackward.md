# aclnnReflectionPad1dBackward

## 支持的产品型号

- 昇腾310P AI处理器。
- 昇腾910 AI处理器。
- 昇腾910B AI处理器。
- 昇腾910_93 AI处理器。

## 接口原型

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnReflectionPad1dBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnReflectionPad1dBackward”接口执行计算。

- `aclnnStatus aclnnReflectionPad1dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclIntArray *padding, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnReflectionPad1dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- 算子功能：reflection_pad1d的反向传播。
- 示例：

  ```
  输入gradOutput([[1, 1, 1, 1, 1]])
  self([[0, 1, 2]])
  padding([1, 1])
  输出为([[1, 3, 1]])
  ```

## aclnnReflectionPad1dBackwardGetWorkspaceSize

- **参数说明：**

  - gradOutput(aclTensor*, 计算输入): 输入的梯度，Device侧的aclTensor。数据类型支持FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128, 数据类型与self一致，支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持NCHW，shape支持2-3维且维度需要与self和gradInput一致，shape需要与reflection_pad1d正向传播的output一致。
  - self(aclTensor*, 计算输入)：需要进行填充的tensor，Device侧的aclTensor。数据类型支持FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128, 支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持NCHW，shape支持2-3维且维度需要与gradOutput和gradInput一致，shape与gradInput一致。
  - padding(aclIntArray*, 计算输入)：填充范围，Device侧的aclIntArray数组，数据类型为INT64，长度为2。padding的两个数值都需小于self最后一维度的数值。
  - gradInput(aclTensor*, 计算输出)：计算得到的self的梯度，数据类型支持FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128, 数据类型与self一致，shape与self一致，支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持NCHW。
  - workspaceSize(uint64_t*, 出参)：返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor**, 出参)：返回op执行器，包含了算子计算流程。
- **返回值：**

  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现如下场景时报错：
  返回161001(ACLNN_ERR_PARAM_NULLPTR)：1. gradOutput, self, padding, gradInput任何一个为空指针。
  返回161002(ACLNN_ERR_PARAM_INVALID)：1. gradOutput、self、padding和gradInput的数据类型或数据格式不在支持的范围之内。
  				     2. gradOutput、self、padding和gradInput的输入shape在支持范围之外。
  				     3. self为空tensor且存在非第一维度的值为0。
  				     4. padding内的数值大于等于self的维度。
  				     5. gradOutput shape需要与reflection_pad1d正向传播的output一致。
  ```

## aclnnReflectionPad1dBackward

- **参数说明：**

  - workspace(void*, 入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnReflectionPad1dBackwardGetWorkspaceSize获取。
  - executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream, 入参)：指定执行任务的AscendCL Stream流。
- **返回值：**

  aclnnStatus：返回状态码。

## 约束与限制

无。
