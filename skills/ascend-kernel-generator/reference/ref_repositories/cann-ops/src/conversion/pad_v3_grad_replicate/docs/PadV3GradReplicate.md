声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# PadV3GradReplicate

## 支持的产品型号

Atlas 训练系列产品/Atlas 推理系列产品/Atlas A2训练系列产品/Atlas 800I A2推理产品/

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 通过aclnn调用的方式调用PadV3GradReplicate算子。

  **说明：**
  无。


## 算子执行接口
每个算子分为两段式接口，必须先调用“aclnnReplicationPad2dBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnReplicationPad2dBackward”接口执行计算。

- `aclnnStatus aclnnReplicationPad2dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclIntArray *padding, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnReplicationPad2dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnReplicationPad2dBackwardGetWorkspaceSize

- **参数说明：**
  
  - gradOutput(aclTensor*，计算输入): Device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128, 数据类型与self一致，支持ND，shape支持3-4维且维度需要与self和gradInput一致，shape需要与replication_pad2d正向传播的output一致。
  - self(aclTensor*，计算输入)：Device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128, 支持ND，shape支持3-4维且维度需要与gradOutput和gradInput一致，shape与gradInput一致。
  - padding(aclIntArray*，计算输入)：Device侧的aclIntArray数组，数据类型为INT64，长度为4。padding前两维度的数值都需小于self最后一维度的数值，后两维度的数值需小于self倒数第二维度的数值。
  - gradInput(aclTensor*，计算输出)：Device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128, 数据类型与self一致，支持ND，shape支持3-4维且与gradOutput和self一致，shape与self一致。
  - workspaceSize(uint64_t*，出参)：返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor**，出参)：返回op执行器，包含了算子计算流程。

- **返回值：**
  ```
  第一段接口完成入参校验，出现如下场景时报错：
  返回161001(ACLNN_ERR_PARAM_NULLPTR)：1. gradOutput, self, padding, gradInput任何一个为空指针。
  返回161002(ACLNN_ERR_PARAM_INVALID)：1. gradOutput、self、padding和gradInput的数据类型或数据格式不在支持的范围之内。
  				     2. gradOutput、self、padding和gradInput的输入shape在支持范围之外。
  				     3. self为空tensor且存在非第一维度的值为0。
  				     4. padding内的数值大于等于self的维度。
  				     5. gradOutput shape需要与replication_pad2d正向传播的output一致。
  ```
### aclnnReplicationPad2dBackward
- **参数说明：**
  
  - workspace(void*，入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t，入参)：在Device侧申请的workspace大小，由第一段接口aclnnReplicationPad2dBackwardGetWorkspaceSize获取。
  - executor(aclOpExecutor*，入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream，入参)：指定执行任务的AscendCL Stream流。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束与限制
- 无
