声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# InplaceFusedMatmulSoftmaxGrad

## 支持的产品型号

- Atlas A2 训练系列产品
- Atlas 800I A2 推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：该InplaceFusedMatmulSoftmaxGrad算子提供等同torch.softmax的反向grad计算功能。InplaceFusedMatmulSoftmaxGrad算子的主要功能是将InplaceAttnSoftmax的输出作为反向传播的输入，并将结果原地写入输入张量。

## 实现原理
先对数据按照rows进行切分，切分完成后进行Matmul计算。
对于cols(输入尾轴大小) <UB(Unified Buffer) :
    调用Ascend C的API接口SoftMaxGrad进行实现。对于16位的数据类型将其通过Cast接口转换为32位浮点数进行计算。
对于cols(输入尾轴大小) >= UB(Unified Buffer) :
    根据softmaxgrad计算公式，调用Ascend C的API接口Mul、Sum、sub、Mul进行实现。对于16位的数据类型将其通过Cast接口转换为32位浮点数进行计算。
具体步骤：
- 输入格式归一化：
- - 将所有输入 reshape 成二维 [rows, cols]，其中 rows 代表前面所有维度的乘积，cols 代表最后一个维度。
这样可以适配不同输入形状，同时保持 softmaxgrad 计算逻辑的一致性。
- 行 (rows) 维度的 Tiling：
- - 分配 40 个核（core），每个核负责 ceil(rows / 40) 行。
- - 首核与尾核优化：
- - - 由于 rows 可能不是 40 的倍数，因此前 rows % 40 个核会多计算一行。
- - -核内 row 维度循环​：每个核负责的 row 在 UB 内部处理，不跨核间传递数据。
- 列 (cols) 维度的 Tiling：
- - 情况 1（cols <= UB 容量）：
- - - 由于 cols 适配单个 UB 空间，因此不需要对 cols 进行额外 tiling。
- - - 直接加载整个 cols 进入 UB 计算，避免多次数据搬运。
- - 情况 2（cols > UB 容量）：
- - - 由于 cols 过大，UB 无法一次性存储完整 cols，需要​在 kernel 内进行 col 维度循环​：
- - - 每次搬运 UB 能承受的 cols 大小（需对 32B 对齐）。

## 算子执行接口

每个算子分为两段式接口，必须先调用“aclnnInplaceFusedMatmulSoftmaxGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnInplaceFusedMatmulSoftmaxGrad”接口执行计算。

* `aclnnStatus aclnnInplaceFusedMatmulSoftmaxGradGetWorkspaceSize(const aclTensor* softmaxOutput, const aclTensor *gradOutput, const aclTensor *values, uint64_t *workspaceSize,aclOpExecutor **executor)`
* `aclnnStatus aclnnInplaceFusedMatmulSoftmaxGrad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnInplaceFusedMatmulSoftmaxGradGetWorkspaceSize

- **参数说明：**
  - softmaxOutput（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入softmaxOutput，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
  - gradOutput（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入softmaxOutput，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
  - values（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入softmaxOutput，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：输入的数据类型和数据格式不在支持的范围内。
  ```

### aclnnInplaceFusedMatmulSoftmaxGrad

- **参数说明：**

  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnInplaceFusedMatmulSoftmaxGradGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。


## 约束与限制

- 输入的数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式只支持ND

## 算子原型
<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="5" align="center">InplaceFusedMatmulSoftmaxGrad</th></tr> 
<tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td><td align="center">约束</td></tr>   
<tr><td rowspan="2" align="center">算子输入</td>
<tr><td align="center">softmaxOutput</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td><td align="center">[m,n]</td></tr>
<tr><td rowspan="2" align="center">算子输入</td>
<tr><td align="center">gradOutput</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td><td align="center">[m,k], k∈[1,65535]</td></tr> 
<tr><td rowspan="2" align="center">算子输入</td> 
<tr><td align="center">values</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td><td align="center">[n,k], k∈[1,65535]</td></tr>   
</table>

## 调用示例

详见[InplaceFusedMatmulSoftmaxGrad自定义算子样例说明算子调用章节](../README.md#算子调用)