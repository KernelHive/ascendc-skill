声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# aclnnAdaptiveAvgPool3dBackward

## 支持的产品型号

Atlas A2训练系列产品/Atlas 800I A2推理产品/Atlas 200I/500 A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：进行AdaptiveAvgPool3d的反向计算。

## 算子执行接口
每个算子分为[两段式接口](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E4%B8%A4%E6%AE%B5%E5%BC%8F%E6%8E%A5%E5%8F%A3.md)，必须先调用“aclnnAdaptiveAvgPool3dBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnAdaptiveAvgPool3dBackward”接口执行计算。

- `aclnnStatus aclnnAdaptiveAvgPool3dBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnAdaptiveAvgPool3dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

## aclnnAdaptiveAvgPool3dBackwardGetWorkspaceSize

- **参数说明：**
  
  - gradOutput（aclTensor*，计算输入）：当前节点的梯度，Device侧的aclTensor。数据类型支持BFLOAT16、FLOAT16、FLOAT32，且数据类型与self一致。支持[非连续的Tensor](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E9%9D%9E%E8%BF%9E%E7%BB%AD%E7%9A%84Tensor.md)，shape支持4维或5维，shape的每一维均为正数，且总维数与self一致。[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持NCDHW、ND，且需要与self数据格式一致。
  - self(aclTensor\*, 计算输入)：输入张量，叶子节点。Device侧的aclTensor，数据类型支持BFLOAT16、FLOAT16、FLOAT32，支持[非连续的Tensor](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E9%9D%9E%E8%BF%9E%E7%BB%AD%E7%9A%84Tensor.md)，shape支持4维或5维，且shape的每一维均为正数。[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持NCDHW、ND。
  - out(aclTensor\*, 计算输出)：输出张量，对应了输入叶子节点的梯度。Device侧的aclTensor，数据类型支持BFLOAT16、FLOAT16、FLOAT32；shape与self保持一致；[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持NCDHW、ND，且与self数据类型一致。
  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR)：1. 传入的gradOutput、self或out是空指针。
  返回161002 (ACLNN_ERR_PARAM_INVALID)：1. gradOutput和self的数据类型和数据格式不在支持的范围之内。
                                        2. gradOutput、self和out数据类型不一致。
                                        3. gradOutput、self和out的维数不等于4或5。
                                        4. gradOutput、self和out的shape不匹配。
                                        5. gradOutput或self的shape的某一维不大于0。
                                        6. gradOutput和self的数据格式不一致。
                                        7. gradOutput的shape不满足单向广播self的shape(仅支持HW维度，NC维度需要保持一致)
  ```

## aclnnAdaptiveAvgPool3dBackward

- **参数说明：**
  
  - workspace（void*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnAdaptiveAvgPool3dBackwardGetWorkspaceSize获取。
  - executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。
- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81.md)。

## 约束与限制
无