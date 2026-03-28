声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# aclnnAvgPool3d

## 支持的产品型号

Atlas A2训练系列产品/Atlas 800I A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：对输入Tensor进行窗口为$kD * kH * kW$、步长为$sD * sH * sW$的三维平均池化操作，其中$k$为kernelSize，表示池化窗口的大小，$s$为stride，表示池化操作的步长。
- 计算公式：
  输入input($N,C,D_{in},H_{in},W_{in}$)、输出out($N,C,D_{out},H_{out},W_{out}$)和池化步长($stride$)、池化窗口大小kernelSize($kD,kH,kW$)的关系是

$$
D_{out}=\lfloor \frac{D_{in}+2*padding[0]-kernelSize[0]}{stride[0]}+1 \rfloor
$$

$$
H_{out}=\lfloor \frac{H_{in}+2*padding[1]-kernelSize[1]}{stride[1]}+1 \rfloor
$$

$$
W_{out}=\lfloor \frac{W_{in}+2*padding[2]-kernelSize[2]}{stride[2]}+1 \rfloor
$$

$$
out(N_i,C_i,d,h,w)=\frac{1}{kD*kH*kW}\sum_{k=0}^{kD-1}\sum_{m=0}^{kH-1}\sum_{n=0}^{kW-1}input(N_i,C_i,stride[0]*d+k,stride[1]*h+m,stride[2]*w+n)
$$

## 算子执行接口
每个算子分为[两段式接口](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E4%B8%A4%E6%AE%B5%E5%BC%8F%E6%8E%A5%E5%8F%A3.md)，必须先调用“aclnnAvgPool3dGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnAvgPool3d”接口执行计算。

- `aclnnStatus aclnnAvgPool3dGetWorkspaceSize(const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride, const aclIntArray* padding, bool ceilMode, bool countIncludePad, int64_t divisorOverride, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`

- `aclnnStatus aclnnAvgPool3d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

## aclnnAvgPool3dGetWorkspaceSize

- **参数说明**：

  - self（aclTensor*,计算输入）：表示待转换的张量，公式中的$input$，Device侧的tensor，支持空tensor场景，数据类型支持FLOAT16、BFLOAT16和FLOAT。支持4维对应的格式为CDHW或5维对应的格式是为NCDHW。支持[非连续的Tensor](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E9%9D%9E%E8%BF%9E%E7%BB%AD%E7%9A%84Tensor.md)。支持[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)为ND。
  - kernelSize（aclIntArray*,计算输入）： 表示池化窗口大小，公式中的$kernelSize$，Host侧的aclIntArray，长度为1（$kD=kH=kW$）或3（$kD, kH, kW$）。数据类型支持INT32和INT64。数值必须大于0且不大于输入对应维度的数值。
  - stride（aclIntArray*,计算输入）： 表示池化操作的步长，公式中的$stride$，Host侧的aclIntArray，长度为0（$默认为kernelSize$）或1（$sD=sH=sW$）或3($sD, sH, sW$)。数据类型支持INT32和INT64。数值必须大于0。
  - padding（aclIntArray*,计算输入）：表示在输入的D、H、W方向上padding补0的层数，公式中的$padding$，Host侧的aclIntArray，长度为1（$padD=padH=padW$）或3（$padD, padH, padW$）。数据类型支持INT32和INT64。数值在[0, kernelSize/2]的范围内。
  - ceilMode（bool，计算输入）： 数据类型支持BOOL。表示计算输出shape时，向下取整（False），否则向上取整。
  - countIncludePad（bool，计算输入）：数据类型支持BOOL。表示平均计算中包括零填充（True），否则不包括。
  - divisorOverride（int64_t，计算输入）：数据类型支持INT64。如果指定，它将用作平均计算中的除数，当值为0时，该属性不生效。
  - out（aclTensor\*，计算输出）： 输出的tensor，公式中的$out$。数据类型支持FLOAT16、BFLOAT16和FLOAT。支持4维（$C,D_{out},H_{out},W_{out}$）或5维（$N,C,D_{out},H_{out},W_{out}$）。支持[非连续的Tensor](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E9%9D%9E%E8%BF%9E%E7%BB%AD%E7%9A%84Tensor.md)。支持[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)为ND。数据类型、[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)需要与self一致。
  - workspaceSize（uint64_t\*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81.md)。

```
第一段接口完成入参校检，出现以下场景时报错：
161001 (ACLNN_ERR_PARAM_NULLPTR)：1. 传入的self、kernelSize、stride、padding或out是空指针。
161002 (ACLNN_ERR_PARAM_INVALID)：1. 传入的self或out的数据类型/数据格式不在支持的范围之内。
                                  2. 传入的self和out的数据类型/数据格式不一致。
                                  3. 传入的kernelSize、stride存在某维度的值小于等于0, padding的值不在[0, kernelSize/2]的范围内。
                                  4. 传入的kernelSize、padding的长度不等于1或者不等于3，stride的长度不等于0或1或3。
                                  5. 根据平均池化语义计算得到的输出shape与接口传入的输出shape不一致。
```

## aclnnAvgPool3d

- **参数说明**：

  - workspace(void \*, 入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnAvgPool3dGetWorkspaceSize获取。
  - executor(aclOpExecutor \*, 入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream, 入参)：指定执行任务的AscendCL Stream流。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81.md)。

## 约束与限制
无