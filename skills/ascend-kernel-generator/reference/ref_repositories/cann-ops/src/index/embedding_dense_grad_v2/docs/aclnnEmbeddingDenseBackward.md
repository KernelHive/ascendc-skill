# aclnnEmbeddingDenseBackward

## 支持的产品型号
- Atlas A2 训练系列产品/Atlas A3 训练系列产品

## 接口原型

每个算子分为两段式接口，必须先调用“aclnnEmbeddingDenseBackwardGetWorkspaceSiz”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnEmbeddingDenseBackward”接口执行计算。

- `aclnnStatus aclnnEmbeddingDenseBackwardGetWorkspaceSize(const aclTensor *grad, const aclTensor *indices, uint64_t numWeights, uint64_t paddingIdx, bool scaleGradByFreq, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnEmbeddingDenseBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`

## 功能描述

算子功能：实现aclnnEmbedding的反向计算, 将相同indice所对应的grad的一行累加到out上


## aclnnEmbeddingDenseBackwardGetWorkspaceSize

- **参数说明：**

  - grad(aclTensor*, 计算输入)：数据的原始梯度，Device侧的aclTensor，支持维度2-8维，除尾轴外合轴后shape与indices合轴后shape相同，支持非连续的Tensor，数据格式支持ND。
    - Atlas A2 训练系列产品/Atlas A3 训练系列产品：数据类型支持BFLOAT16、FLOAT16、FLOAT。
  - indices(aclTensor*, 计算输入)：grad输入对应的索引值，Device侧的aclTensor, 取值范围为[0, numWeights)，支持维度1-8维, 支持非连续的Tensor，数据格式支持ND。数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL。
  - numWeights(uint64_t, 计算输入)：输出tensor的首轴大小。
  - paddingIdx(uint64_t, 计算输入)：将输出tensor中第paddingIdx行填充成0，如果paddingIdx为负数则不进行处理。
  - scaleGradByFreq(bool, 计算输入)：根据单词出现的频率，对梯度进行放缩，若为true，则对结果按词频进行缩放，若为false，则不进行处理。
  - out(aclTensor*, 计算输出)：梯度求和的结果输出，Device侧的aclTensor，维度为2维，首轴大小为numWeights，尾轴大小与grad尾轴相同，数据类型与grad类型相同，数据格式仅支持ND。
    - Atlas A2 训练系列产品/Atlas A3 训练系列产品：数据类型支持BFLOAT16、FLOAT16、FLOAT。
  - workspaceSize(uint64_t *, 出参): 返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor **, 出参): 返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见aclnn返回码。

  ```
  第一段接口完成入参校验，出现如下场景时报错：
  返回161001(ACLNN_ERR_PARAM_NULLPTR)：1. 传入的grad、indices、out是空指针。
  返回161002(ACLNN_ERR_PARAM_INVALID)：1. grad、indices、out的数据类型和数据格式不在支持的范围之内。
                                      2. grad, indices的维度超过8维
                                      3. grad与indices的shape不满足约束条件
                                      4. out的shape不符合推导结果

  ```

## aclnnEmbeddingDenseBackward


- **参数说明：**

  * workspace(void *, 入参): 在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t, 入参): 在Device侧申请的workspace大小，由第一段接口aclnnEmbeddingDenseBackwardGetWorkspaceSize获取。
  * executor(aclOpExecutor *, 入参): op执行器，包含了算子计算流程。
  * stream(aclrtStream, 入参): 指定执行任务的AscendCL Stream流。
  
- **返回值：**

  aclnnStatus：返回状态码，具体参见aclnn返回码。

## 约束与限制
- Atlas A2 训练系列产品/Atlas A3 训练系列产品
  - 在参数shape超过以下限制时，输出无法保证高精度，若开启了确定性计算，也无法保证高性能
    - grad合轴成二维shape后，第一个维度超过INT32_MAX(2147483647)
    - numWeights超过INT32_MAX(2147483647)
  - indices合轴后维度超过INT32_INF(2139095040)时，无法保证高性能

## 算子原型
<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">EmbeddingDenseGradV2</td></tr>
<tr><td align="center"></td><td align="center">name</td><td align="center">type</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td rowspan="7" align="center">算子输入</td>
<tr><td align="center">grad</td><td align="center">tensor</td><td align="center">float32</td><td align="center">ND</td></tr>
<tr><td align="center">sort_indices</td><td align="center">tensor</td><td align="center">int32</td><td align="center">ND</td></tr>
<tr><td align="center">pos_idx</td><td align="center">tensor</td><td align="center">int32</td><td align="center">ND</td></tr>
<tr><td align="center">num_weights</td><td align="center">tensor</td><td align="center">int</td><td align="center">-</td></tr>
<tr><td align="center">padding_idx</td><td align="center">tensor</td><td align="center">int</td><td align="center">-</td></tr>
<tr><td align="center">scale_grad_by_freq</td><td align="center">tensor</td><td align="center">bool</td><td align="center">-</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">tensor</td><td align="center">float32</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">embedding_dense_grad_v2</td></tr>
</table>