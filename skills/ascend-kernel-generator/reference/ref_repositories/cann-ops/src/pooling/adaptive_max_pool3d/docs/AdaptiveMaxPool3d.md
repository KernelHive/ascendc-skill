声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# AdaptiveMaxPool3d

## 支持的产品型号

A2训练系列产品/Atlas

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能： 根据输入的outputSize计算每次kernel的大小，对输入self进行3维最大池化操作，输出池化后的值out和索引indices。aclnnAdaptiveMaxPool3d与aclnnMaxPool3d的区别在于，只需指定outputSize大小，并按outputSize的大小来划分pooling区域。

- 计算公式
  - out tensor 中对于DHW轴上每个位置为$(l,m,n)$的元素来说，其计算公式为：
  $$ D^{l}_{left} = floor((l*D)/D_o)$$ $$ D^{l}_{right} = ceil(((l+1)*D)/D_o)$$ $$ H^{m}_{left} = floor((m*H)/H_o)$$ $$ H^{m}_{right} = ceil(((m+1)*H)/H_o)$$ $$ W^{n}_{left} = floor((n*W)/W_o)$$ $$ W^{n}_{right} = ceil(((n+1)*W)/W_o)$$ $$out(N,C,l,m,n)=\underset {i \in [D^{l}_{left}, D^{l}_{right}],j\in [H^m_{left},H^m_{right}], k \in [W^n_{left},W^n_{right}] }{max} input(N,C,i,j,k)$$ $$indices(N,C,l,m,n)=\underset {i \in [D^{l}_{left}, D^{l}_{right}],j\in [H^m_{left},H^m_{right}], k \in [W^n_{left},W^n_{right}] }{argmax} input(N,C,i,j,k)$$

- Shape描述：
  - self.shape = (N, C, Din, Hin, Win)
  - outputSize = [Din, Hout, Wout]
  - out.shape = (N, C, Din, Hout, Wout)
  - indices.shape = (N, C, Din, Hout, Wout)
  
  **说明：**
  无。

## 实现原理

## 算子执行接口

每个算子分为两段式接口，必须先调用“aclnnAdaptiveMaxPool3dGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnAdaptiveMaxPool3d”接口执行计算。

- `aclnnStatus aclnnAdaptiveMaxPool3dGetWorkspaceSize(const aclTensor* self, const aclIntArray* outputSize, aclTensor* out, aclTensor* indices, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnAdaptiveMaxPool3d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnAdaptiveMaxPool3dGetWorkspaceSize

- **参数说明：**
  
  - self（aclTensor*，计算输入）：输入Tensor，Device侧的aclTensor。shape支持5D。支持[非连续的Tensor]，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持NCDHW。D轴H轴W轴3个维度的乘积$D*H*W$不能大于int32的最大表示,数据类型支持BFLOAT16、FLOAT16、FLOAT32，且数据类型与out的数据类型保持一致。
  - outputSize（aclIntArray\*，计算输入）：Host侧的aclIntArray，size大小为3。表示输出结果在$D_o$，$H_o$，$W_o$维度上的空间大小。数据类型支持INT32和INT64。outputSize中元素值不可小于0。
  - out（aclTensor\*，计算输出）：输出Tensor，是Device侧的aclTensor。池化后的结果。shape与indices保持一致。[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持NCDHW,数据类型支持BFLOAT16、FLOAT16、FLOAT32，且数据类型与self的数据类型一致。
  - indices（aclTensor\*，计算输出）：输出Tensor，是Device侧的aclTensor。indices表示out元素在输入self中的索引位置。shape与out保持一致，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持NCDHW,数据类型支持INT32.
  - workspaceSize（uint64_t*，出参）：返回用户需要在npu Device侧申请的workspace大小。
  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。
  
  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR)：1. 传入的self、outputSize、out或indices是空指针。
  返回161002 (ACLNN_ERR_PARAM_INVALID)：1. self、out、indices的数据类型、shape、format、参数取值不在支持的范围之内。
                                        2. outputSize的shape、参数取值不在支持的范围内
                                        3. self和out数据类型不一致
                                        4. out和indices shape不一致
                                        5. 平台不支持
  ```

### aclnnAdaptiveMaxPool3d

- **参数说明：**
  
  - workspace（void*，入参）：在npu device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在npu device侧申请的workspace大小，由第一段接口aclnnAdaptiveMaxPool3dGetWorkspaceSize获取。
  - executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束与限制

- 无

## 调用示例

详见[AdaptiveMaxPool3d自定义算子样例说明算子调用章节](../README.md#算子调用)
