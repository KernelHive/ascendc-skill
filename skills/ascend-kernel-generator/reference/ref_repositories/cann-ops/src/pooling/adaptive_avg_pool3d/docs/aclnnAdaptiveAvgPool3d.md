# aclnnAdaptiveAvgPool3d
## 支持的产品型号
- Atlas 推理系列产品
- Atlas A2 训练系列产品/Atlas 800I A2 推理产品

## 接口原型
每个算子分为两段式接口，必须先调用“aclnnAdaptiveAvgPool3dGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnAdaptiveAvgPool3d”接口执行计算。

- `aclnnStatus aclnnAdaptiveAvgPool3dGetWorkspaceSize(const aclTensor* self, const aclIntArray* outputSize, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnAdaptiveAvgPool3d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## 功能描述

算子功能：在指定三维输出shape信息（outputSize）的情况下，完成张量self的3D自适应平均池化计算。aclnnAdaptiveAvgPool3d与aclnnAvgPool3d不同的是，aclnnAdaptiveAvgPool3d只需要指定输出的大小，就可以自动推导出kernel的大小与对应的步长。

- Shape描述：
  - self.shape = (N, C, Din, Hin, Win) 或者 (C, Din, Hin, Win)
  - outputSize = [Dout, Hout, Wout]
  - out.shape = (N, C, Dout, Hout, Wout) 或者 (C, Dout, Hout, Wout)

## aclnnAdaptiveAvgPool3dGetWorkspaceSize

- **参数说明：**
  
  - self（aclTensor*，计算输入）：表示待计算的目标张量，Device侧的aclTensor。支持非连续的Tensor，不支持空tensor，shape支持4-5维。数据格式支持NCDHW、ND。
    - 昇腾310P AI处理器：数据类型支持FLOAT16、FLOAT32。
    - 昇腾910B AI处理器：数据类型支持BFLOAT16、FLOAT16、FLOAT32。

  - outputSize(aclIntArray\*, 计算输入)：指定输出在DHW维度上的shape大小，Device侧的aclIntArray。数据类型支持INT32和INT64，数组长度恒为3。
  - out(aclTensor\*, 计算输出)：Device侧的aclTensor，与self的数据类型一致；out的shape需要与self的shape和outputSize推导出的shape结果一致。支持非连续的Tensor。数据格式支持NCDHW、ND，与self一致。
    - 昇腾310P AI处理器：数据类型支持FLOAT16、FLOAT32。
    - 昇腾910B AI处理器：数据类型支持BFLOAT16、FLOAT16、FLOAT32。

  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**

  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR)：1. 传入的self、outputSize或out是空指针。
  返回161002 (ACLNN_ERR_PARAM_INVALID)：1. self的数据类型和数据格式不在支持的范围之内。
                                        2. self和out数据类型不一致。
                                        3. self的维度不等于4或5。
                                        4. outputSize长度不为3。
                                        5. out的shape与self的shape和outputSize推导出的shape结果不一致
  ```

## aclnnAdaptiveAvgPool3d

- **参数说明：**
  
  - workspace（void*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnAdaptiveAvgPool3dGetWorkspaceSize获取。
  - executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。
- **返回值：**

  aclnnStatus：返回状态码。

## 约束与限制
无
