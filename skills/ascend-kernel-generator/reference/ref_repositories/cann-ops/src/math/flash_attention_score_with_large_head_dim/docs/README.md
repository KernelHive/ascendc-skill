声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# FlashAttentionScoreWithLargeHeadDim

## 支持的产品型号

Atlas A2 训练系列产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。

- 计算公式：

    注意力的正向计算公式如下：

    $$
    attention\_out = Dropout(Softmax(Mask(scale*(pse+query*key^T), atten\_mask)), keep\_prob)*value
    $$

## 实现原理

图1 训练计算流程图

![FA图](./fig/FlashAttentionScoreWithLargeHeadDim.png)

按照flashAttention正向计算流程实现，整体计算流程如下：

1. query与转置后的key做matmul计算后得到最初步的attention_score，然后与位置编码pse相加后再乘以缩放系数scale_value。此时的结果通过atten_mask进行select操作，将atten_mask中为true的位置进行遮蔽，得到结果masked_attention_score，即atten_mask中为true的位置在select后结果为负的极小值，经过softmax计算之后变成0从而达到遮蔽效果。

2. 为了实现FlashAttention加速算法，使用FlashSoftmax操作对masked_attention_score进行运算，用以代替原公式中的softmax运算，而后将结果与value做matmul运算。由于FlashSoftmax操作对masked_attention_score的Skv(输入key、value的sequence length)方向进行了切分，故实现过程中存在一个刷新流程，具体如下：
    1. 每次FlashSoftmax计算只对切分后的一个SkvSplit（SkvSplit是针对Skv轴进行切分之后的序列长度的简称）进行操作，并从第二次循环开始记录exp，其中 i 表示Skv切分后的循环变量，针对exp的i是从1开始 ，exp的计算公式如下：
       $$
       exp[i] = e^{max_{i - 1} - max_{i}}
       $$
    2. 当i = 0时，计算出的MM[PV]结果直接保存到ub_attention_out[0]的ub中。
    3. 从i = 1开始，需要增加Mul和Add操作，即将上一次的MM[PV]的结果和当前exp相乘，相乘完的结果和本次MM[PV]的结果相加得到的结果保存到ub_attention_out[1]的ub中。以此类推，遍历Skv计算完成。
    4. 由于FlashSoftmax计算中的除sum被后移到输出attention_out之前，因此最后需要将ub中的ub_attention_out按行除以softmax_sum并将最终完整的结果保存到输出内存attention_out(Final)上。

## 算子执行接口

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnFlashAttentionScoreWithLargeHeadDimGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnFlashAttentionScoreWithLargeHeadDim”接口执行计算。

* `aclnnStatus aclnnFlashAttentionScoreWithLargeHeadDimGetWorkspaceSize(const aclTensor *query, const aclTensor *key, const aclTensor *value, double scaleValueOptional, int64_t headNum, const aclTensor *softmaxMaxOut, const aclTensor *softmaxSumOut, const aclTensor *attentionOutOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnFlashAttentionScoreWithLargeHeadDim(void *workspace, int64_t workspaceSize, aclOpExecutor **executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnFlashAttentionScoreWithLargeHeadDimGetWorkspaceSize

- **参数说明：**

  - query（aclTensor\*，计算输入）：Device侧的aclTensor，数据类型支持FLOAT16，数据类型与key/value的数据类型一致，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - key（aclTensor\*，计算输入）：Device侧的aclTensor，数据类型支持FLOAT16，数据类型与query/value的数据类型一致，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - value（aclTensor\*，计算输入）：Device侧的aclTensor，数据类型支持FLOAT16，数据类型与query/key的数据类型一致，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - scaleValueOptional（double，计算输入）：Host侧的double，可选参数，公式中的scale，代表缩放系数，作为计算流中Muls的scalar值，数据类型支持DOUBLE，一般设置为D^-0.5。
  - headNum（int64\_t，计算输入）：Host侧的int64_t，代表单卡的head个数，即输入query的N轴长度，数据类型支持INT64。
  - softmaxMaxOut（aclTensor\*，计算输出）：Device侧的aclTensor，Softmax计算的Max中间结果，用于反向计算。数据类型支持FLOAT，输出的shape类型为[B,N,Sq,8]，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - softmaxSumOut（aclTensor\*，计算输出）：Device侧的aclTensor，Softmax计算的Sum中间结果，用于反向计算。数据类型支持FLOAT，输出的shape类型为[B,N,Sq,8]，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - softmaxOutOut（aclTensor\*，计算输出）：预留参数，暂未使用。
  - attentionOutOut（aclTensor\*，计算输出）：Device侧的aclTensor，计算公式的最终输出。数据类型支持FLOAT16，数据类型和shape与query一致，输出的shape类型为[B,N,Sq,D]，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - workspaceSize（uint64\_t\*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
  
- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：query、key、value、softmaxMaxOut、softmaxSumOut、softmaxOutOut、attentionOutOut的数据类型和数据格式不在支持的范围内。
  ```

### aclnnFlashAttentionScoreWithLargeHeadDim

- **参数说明：**

  -   workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  -   workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnFlashAttentionScoreWithLargeHeadDimGetWorkspaceSize获取。
  -   executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  -   stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束与限制

- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
- 输入query、key、value的B：batchsize必须相等。
- 输入query、key、value的D：Head-Dim必须相等。
- 输入query、key、value的input_layout必须一致。
- 输入query、key、value的数据类型必须一致。
- 输入key/value的shape必须一致。
- 支持输入query的N和key/value的N不相等，但必须成比例关系，即Nq/Nkv必须是非0整数，Nq取值范围1~256。当Nq/Nkv > 1时，即为GQA(grouped-query attention)；当Nkv=1时，即为MQA(multi-query attention)。本文如无特殊说明，N表示的是Nq。
- 关于数据shape的约束，以inputLayout的BSND、BNSD为例（BSH、SBH下H=N\*D），其中：
    - B：取值范围为1\~2M。
    - N：取值范围为1\~256。
    - S：取值范围为1\~1M。
    - D：取值范围为1\~512。
- 部分场景下，如果计算量过大可能会导致算子执行超时(aicore error类型报错，errorStr为：timeout or trap error)，此时建议做轴切分处理，注：这里的计算量会受B、S、N、D等参数的影响，值越大计算量越大。
## 算子原型

```c++
REG_OP(FlashAttentionScoreWithLargeHeadDim)
    .INPUT(query, TensorType({DT_FLOAT16}))
    .INPUT(key, TensorType({DT_FLOAT16}))
    .INPUT(value, TensorType({DT_FLOAT16}))
    .OUTPUT(softmax_max, TensorType({DT_FLOAT32}))
    .OUTPUT(softmax_sum, TensorType({DT_FLOAT32}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16}))
    .ATTR(scale_value, Float, 1.0)
    .REQUIRED_ATTR(head_num, Int)
    .OP_END_FACTORY_REG(FlashAttentionScoreWithLargeHeadDim)
```

## 调用示例

详见[FlashAttentionScoreWithLargeHeadDim自定义算子样例说明算子调用章节](../README.md#算子调用)