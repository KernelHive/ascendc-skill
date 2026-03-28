
# aclnnGridSampler2DBackward

## 支持的产品型号

- Atlas 训练系列产品。
- Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件。
- Atlas A3 训练系列产品/Atlas A3 推理系列产品。

## 接口原型

每个算子分为两段式接口，必须先调用“aclnnGridSampler2DBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGridSampler2DBackward”接口执行计算。

- `aclnnStatus aclnnGridSampler2DBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* input, const aclTensor* grid, int64_t interpolationMode, int64_t paddingMode, bool alignCorners, const aclBoolArray* outputMask, aclTensor* inputGrad, aclTensor* gridGrad, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnGridSampler2DBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## 功能描述

算子功能：aclnnGridSampler2D的反向传播，完成张量input与张量grid的梯度计算。

## aclnnGridSampler2DBackwardGetWorkspaceSize

- **参数说明：**

  - gradOutput（aclTensor*，计算输入）：表示反向传播过程中上一层的输出梯度，Device侧的aclTensor。数据类型需要与input保持一致，支持非连续的Tensor，不支持空Tensor。shape仅支持四维。数据格式支持ND。
    - Atlas 训练系列产品：数据类型支持FLOAT16、FLOAT32。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持BFLOAT16、FLOAT16、FLOAT32、DOUBLE。
  - input（aclTensor*，计算输入）：表示输入张量，Device侧的aclTensor。支持非连续的Tensor，不支持空Tensor。shape仅支持四维，且需满足input的第一个维度和grid、gradOutput的第一个维度值相同，input的第二个维度和gradOutput的第二个维度值相同，input最后两维的维度值不可为0。数据格式支持ND。
    - Atlas 训练系列产品：数据类型支持FLOAT16、FLOAT32。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持BFLOAT16、FLOAT16、FLOAT32、DOUBLE。
  - grid（aclTensor*，计算输入）：表示采用像素位置的张量，Device侧的aclTensor。数据类型需要与input保持一致，支持非连续的Tensor，不支持空Tensor。shape仅支持四维，且需满足grid的第二个维度和gradOutput的第三个维度值相同，grid的第三个维度和gradOutput的第四个维度值相同，grid最后一维的值等于2。数据格式支持ND。
    - Atlas 训练系列产品：数据类型支持FLOAT16、FLOAT32。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持BFLOAT16、FLOAT16、FLOAT32、DOUBLE。
  - interpolationMode（int64_t，计算输入）：用于表示插值模式，Host侧的整型。0：bilinear（双线性插值），1：nearest（最邻近插值）。
  - paddingMode（int64_t，计算输入）：用于表示填充模式，Host侧的整型。即当（x,y）取值超过输入特征图采样范围时，返回一个特定值，有0：zeros、1：border两种模式。
  - alignCorners（bool，计算输入）：用于表示设定特征图坐标与特征值的对应方式，Host侧的bool类型。设定为true时，特征值位于像素中心。
  - outputMask（aclBoolArray，计算输入）：用于表示输出的掩码，Host侧的aclBoolArray类型。outputMask[0]为true/false，表示 是/否 获取输出inputGrad；outputMask[1]为true/false，表示 是/否 获取输出gridGrad。
  - inputGrad（aclTensor*，计算输出）：表示反向传播的输出梯度，Device侧的aclTensor。数据类型需要与input保持一致，shape需要与input保持一致，支持非连续的Tensor。数据格式支持ND。
    - Atlas 训练系列产品：数据类型支持FLOAT16、FLOAT32。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持BFLOAT16、FLOAT16、FLOAT32、DOUBLE。
  - gridGrad（aclTensor*，计算输出）：表示grid梯度，Device侧的aclTensor。数据类型需要与input保持一致，shape需要与grid保持一致，支持非连续的Tensor。数据格式支持ND。
    - Atlas 训练系列产品：数据类型支持FLOAT16、FLOAT32。
    - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持BFLOAT16、FLOAT16、FLOAT32、DOUBLE。
  - workspaceSize（uint64_t*，出参）: 返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**，出参）: 返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）: 传入的gradOutput、input、grid、inputGrad、gridGrad是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）: 1. gradOutput、input、grid、inputGrad、gridGrad的数据类型不在支持的范围之内。
                                        2. gradOutput、input、grid的shape维度值不为四维。
                                        3. interpolationMode或paddingMode的值不在支持范围内。
                                        4. input的第一个维度和grid、gradOutput的第一个维度值不相同。
                                        5. input的第二个维度和gradOutput的第二个维度值不相同。
                                        6. grid的第二个维度和gradOutput的第三个维度值不相同。
                                        7. grid的第三个维度和gradOutput的第四个维度值不相同。
                                        8. input最后两维的维度值为0。
                                        9. grid最后一维的值不等于2。
                                        10. input的shape和inputGrad的shape不一致，或grid的shape和gridGrad的shape不一致。
  ```

## aclnnGridSampler2DBackward

- **参数说明：**

  - workspace（void*，入参）: 在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）: 在Device侧申请的workspace大小，由第一段接口aclnnGridSampler2DBackwardGetWorkspaceSize获取。
  - executor（aclOpExecutor*，入参）: op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）: 指定执行任务的AscendCL Stream流。
- **返回值：**

  aclnnStatus：返回状态码。

## 约束与限制

无。
