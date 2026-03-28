声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# ReflectionPad3dGrad

## 支持的产品型号

/Atlas 800I A2推理产品/

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 通过aclnn调用的方式调用PadV3GradReplicate算子。

  **说明：**
  无。

## 实现原理



## 算子执行接口

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnReflectionPad3dGradGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnReflectionPad3dGrad”接口执行计算。

* `aclnnStatus aclnnReplicationPad3dBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self, const aclIntArray* padding, aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor)`

* `aclnnStatus aclnnReplicationPad3dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnReflectionPad3dGradGetWorkspaceSize

- **参数说明：**
  
  - gradOutput（aclTensor*，计算输入）：反向传播的输入，Device侧的aclTensor。维度支持四维或五维且与self和gradInput一致，shape、dtype需要与正向传播aclnnReplicationPad3d的输出out一致。
    - 昇腾A2 AI处理器：数据类型支持FLOAT16、FLOAT32、BFLOAT16、DOUBLE、COMPLEX64、COMPLEX128。
  - self（aclTensor*，计算输入）：正向的输入张量，Device侧的aclTensor。维度支持四维或五维且与gradOutput和gradInput一致，shape、dtype需要与gradInput一致。
    - 昇腾A2 AI处理器：数据类型支持FLOAT16、FLOAT32、BFLOAT16、DOUBLE、COMPLEX64、COMPLEX128。
  - padding（aclIntArray*，计算输入）：Device侧的aclIntArray数组，长度为6，数值依次代表左右上下前后需要填充的值。padding前两个数值需小于self最后一维度的数值，中间两个数值需小于self倒数第二维度的数值，后两个数值需小于self倒数第三维度的数值。
  - gradInput（aclTensor*，计算输出）：反向传播的输出，Device侧的aclTensor。维度支持四维或五维且与gradOutput和self一致，dtype需要与gradOutput一致。
    - 昇腾A2 AI处理器：数据类型支持FLOAT16、FLOAT32、BFLOAT16、DOUBLE、COMPLEX64、COMPLEX128。
  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**
    ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001(ACLNN_ERR_PARAM_NULLPTR)：1. Tensor为空指针。
  返回161002(ACLNN_ERR_PARAM_INVALID)：1. gradOutput、self、padding和gradInput的数据类型或数据格式不在支持的范围之内。
                                      2. gradOutput、self、padding和gradInput的输入shape在支持范围之外。
                                      3. self为空tensor且存在非第一维度的值为0。
                                      4. padding内的数值大于等于self的维度。
                                      5. gradOutput shape需要与replication_pad3d正向传播的output一致。
  ```
### aclnnReflectionPad3dGrad

- **参数说明：**
  
  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnReflectionPad3dGradGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束与限制
- 无
