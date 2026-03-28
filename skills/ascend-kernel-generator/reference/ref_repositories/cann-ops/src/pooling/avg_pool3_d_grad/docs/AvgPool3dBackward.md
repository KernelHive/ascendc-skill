声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# aclnnAvgPool3dBackward

## 支持的产品型号

Atlas A2训练系列产品/Atlas 800I A2推理产品/Atlas 200I/500 A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：三维平均池化的反向传播，计算三维平均池化正向传播的输入梯度。

- 计算公式：

$$
D_{in} = (D_{out} - 1) * {stride[0]} + kernel\_size[0] - 2 * padding[0]
$$
$$
H_{in} = (H_{out} - 1) * {stride[1]} + kernel\_size[1] - 2 * padding[1]
$$
$$
W_{in} = (W_{out} - 1) * {stride[2]} + kernel\_size[2] - 2 * padding[2]
$$

## 算子执行接口
每个算子分为两段式接口，必须先调用“aclnnAvgPool3dBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnAvgPool3dBackward”接口执行计算。

- `aclnnStatus aclnnAvgPool3dBackwardGetWorkspaceSize(const aclTensor* gradOuput, const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride, const aclIntArray* padding, bool ceilMode, bool countIncludePad, int64_t divisorOverride, aclTensor* output, uint64_t* workspaceSize, aclOpExecutor** executor)`

- `aclnnStatus aclnnAvgPool3dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

## aclnnAvgPool3dBackwardGetWorkspaceSize

- **参数说明**：
  - gradOutput(aclTensor*, 计算输入)：输入梯度，npu device侧的aclTensor，支持4维（C, D, H, W）或者5维(N, C, D, H, W), 支持[非连续的Tensor](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E9%9D%9E%E8%BF%9E%E7%BB%AD%E7%9A%84Tensor.md),[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持BFLOAT16、FLOAT16、FLOAT32
  - self(aclTensor*, 计算输入)：输入数据，npu device侧的aclTensor，支持4维（C, D, H, W）或者5维(N, C, D, H, W), 支持[非连续的Tensor](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E9%9D%9E%E8%BF%9E%E7%BB%AD%E7%9A%84Tensor.md),[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持BFLOAT16、FLOAT16、FLOAT32
  - kernelSize（aclIntArray*,计算输入）: Host侧的aclIntArray，表示池化窗口大小，INT32或者INT64类型数组，长度为1(KD = KH = KW)或3(KD, KH, KW)。数值必须大于0.
  - stride（aclIntArray*,计算输入）: Host侧的aclIntArray，表示池化操作的步长，INT32或者INT64类型数组，长度为0(数值与kernelSize数值保持一致)或者1(SD = SH = SW)或者3(SD, SH, SW)。数值必须大于0.
  - padding（aclIntArray*,计算输入）: Host侧的aclIntArray，表示在输入的D, H、W方向上padding补0的层数，INT32或者INT64类型数组，长度为1(PD = PH = PW)或3(PD, PH, PW)。数值在[0, kernelSize/2]的范围内。
  - ceilMode（bool，计算输入）: 表示正向平均池化过程中推导的输出的shape是否向上取整(True)。数据类型支持BOOL。
  - countIncludePad（bool，计算输入）: 计算正向平均池化时是否包括padding填充的0(True)。数据类型支持BOOL。
  - divisorOverride（int64_t，计算输入）: 表示取平均的除数。如果指定，它将用作平均计算中的除数，当值为0时，该属性不生效，数据类型支持INT64。
  - output（aclTensor *，计算输出）: Device侧的aclTensor。支持[非连续的Tensor](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E9%9D%9E%E8%BF%9E%E7%BB%AD%E7%9A%84Tensor.md)。支持4维（C, D, H, W）或者5维(N, C, D, H, W), 支持[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)为ND。数据类型、[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)需要与gradOutput一致。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持BFLOAT16、FLOAT16、FLOAT。
  - workspaceSize（uint64_t \*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81.md)。

  ```
  第一段接口完成入参校检，出现以下场景时报错：
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. 传入的gradOutput、self、kernelSize、padding或output是空指针。
  161002 (ACLNN_ERR_PARAM_INVALID): 1. 传入的gradOutput、self和output的数据类型/维度不在支持的范 围之内。
                                    2. 传入kernelSize，stride, padding的维度不在支持的范围之内。
                                    3. 传入的kernelSize、stride, padding某个维度值小于0。
                                    4. 属性padding超过kernelSize对应位置的1/2,例如paddingH=2,kernelSizeH=2,paddingH>kernelSizeH*1/2。
                                    5. output维度与selfshape不一致
  ```

## aclnnAvgPool3dBackward

- **参数说明**：

  - workspace(void\*, 入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnAvgPool3dBackwardGetWorkspaceSize获取。
  - executor(aclOpExecutor\*, 入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream, 入参)：指定执行任务的AscendCL Stream流。

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81.md)。

## 约束与限制
无