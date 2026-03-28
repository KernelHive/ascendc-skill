声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# aclnnAdaptiveMaxPool3dBackward

## 支持的产品型号

A2训练系列产品/Atlas

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：
  正向自适应最大池化的反向传播，将梯度回填到每个自适应窗口最大值的坐标处，相同坐标处累加。
- 正向计算公式：
  对于输入self维度$[N,C,D,H,W]$，outputSize值为$[D_o,H_o,W_o]$的场景，其输出output维度为$[N,C,D_o,H_o,W_o]$，索引indices维度为$[N,C,D_o,H_o,W_o]$，相应tensor中每个元素的计公式如下：
  $$
  D_{left}^l = \lfloor(l*D)/D_o\rfloor \\
  D_{right}^l = \lceil(l*D)/D_o\rceil \\
  H_{left}^m = \lfloor(m*H)/H_o\rfloor \\
  H_{right}^m = \lceil(m*H)/H_o\rceil  \\
  W_{left}^n = \lfloor(n*W)/W_o\rfloor \\
  W_{right}^n = \lceil(n*W)/W_o\rceil  \\
  output(N,C,l,m,n) = \mathop{\max}\limits_{i \in [D_{left}^l,D_{right}^l],j\in[H_{left}^m,H_{right}^m],k\in[W_{left}^n,W_{right}^n]} input(N,C,i,j,k) \\
  indices(N,C,l,m,n) = \mathop{\arg\max}\limits_{i \in [D_{left}^l,D_{right}^l],j\in[H_{left}^m,H_{right}^m],k\in[W_{left}^n,W_{right}^n]} input(N,C,i,j,k)
  $$
  
  **说明：**
  无。

## 实现原理

## 算子执行接口

每个算子分为两段式接口，必须先调用“aclnnAdaptiveMaxPool3dBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnAdaptiveMaxPool3dBackward”接口执行计算。

- `aclnnStatus aclnnAdaptiveMaxPool3dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *indices, aclTensor *gradInput， uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnAdaptiveMaxPool3dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnAdaptiveMaxPool3dBackwardGetWorkSpaceSize

- **参数说明：**
  
 - gradOutput(aclTensor \*, 计算输入): 梯度Tensor，Device侧aclTensor。和正向的输出shape一致。支持非连续的Tensor，数据格式支持ND, 当输入是5维时，内部按照NCDHW处理，当输入是4维时，在0维度处补1，内部按照NCDHW处理。
    
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  - self(aclTensor \*, 计算输入): 正向的输入Tensor，Device侧aclTensor。支持非连续的Tensor，数据格式支持ND, 当输入是5维时，内部按照NCDHW处理，当输入是4维时，在0维度处补1，内部按照NCDHW处理，与gradOutput一致。
    
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  - indices(aclTensor \*, 计算输入): 输入Tensor，是Device侧aclTensor。正向输入中最大元素的索引位置。数据格式与gradOutput保持一致。shape与gradOutput一致
    
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型仅支持INT32。
  - gradInput(aclTensor \*, 计算输出): 反向输出Tensor，是Device侧aclTensor。shape与self保持一致。支持非连续的Tensor，数据格式与self保持一致。
    
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  - workspaceSize(uint64_t \*, 出参): 返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor \*\*, 出参): 返回op执行器，包含了算子计算流程。
- **返回值：**
  
  aclnnStatus: 返回状态码，具体参见aclnn返回码。
  
  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001(ACLNN_ERR_PARAM_NULLPTR)：1. 传入的gradOutput、self或indices是空指针。
  161002(ACLNN_ERR_PARAM_INVALID)：1. gradOutput、self、indices、gradInput的数据类型不在支持的范围内。
                                   2. gradOutput、self、indices、gradInput的数据格式不在支持的范围内。
                                   3. 输入输出的shape不是4维或者5维。
                                   4. gradOutput与indices的shape不一致，self和gradInput的shape不一致。
                                   5. depth * height * width > max int32，超出了indices的表达范围。
  ```

### aclnnAdaptiveMaxPool3dBackward

- **参数说明：**
  
  - workspace（void*，入参）：在npu device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在npu device侧申请的workspace大小，由第一段接口aclnnAdaptiveMaxPool3DGradGetWorkspaceSize获取。
  - executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束与限制

- 无

## 调用示例

详见[AdaptiveMaxPool3dBackward自定义算子样例说明算子调用章节](../README.md#算子调用)
