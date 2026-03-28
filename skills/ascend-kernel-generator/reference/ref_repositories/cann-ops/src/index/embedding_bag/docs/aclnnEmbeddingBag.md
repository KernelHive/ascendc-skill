# aclnnEmbeddingBag

## 支持的产品型号

- Atlas 推理系列产品
- Atlas 训练系列产品
- Atlas A2 训练系列产品/Atlas A3 训练系列产品

## 接口原型
每个算子分为两段式接口，必须先调用“aclnnEmbeddingBagGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnEmbeddingBag”接口执行计算。
  - `aclnnStatus aclnnEmbeddingBagGetWorkspaceSize(const aclTensor* weight, const aclTensor* indices,const aclTensor* offsets, bool scaleGradByFreq,int64_t mode, bool sparse, const aclTensor* perSampleWeights, bool includeLastOffset, int64_t paddingIdx, aclTensor* output, aclTensor* offset2bag, aclTensor* bagSize, aclTensor* maxIndices, uint64_t* workspaceSize, aclOpExecutor** executor)`
  - `aclnnStatus aclnnEmbeddingBag(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## 功能描述

- 算子功能：根据indices从weight中获得一组被聚合的数，然后根据offsets的偏移和mode指定的聚合模式对获取的数进行max、sum、mean聚合。其余参数则更细化了计算过程的控制。
- shape推导方式如下：
  假设:
    > weight的shape为(numWeight, embeddingDim)
    > indices的shape为(bagIndices, wordsIndices)
    > offsets的shape为(bagOffsets, wordsOffsets)
  - 当mode为sum模式：
    > output的shape 为 includeLastOffset ? (bagOffsets \* wordsOffsets - 1, embeddingDim) : (bagOffsets \* wordsOffsets, embeddingDim)
    > offset2bag的shape 为 (0,) 或 (bagIndices,)
    > bagSize的shape 为 includeLastOffset ? (bagOffsets - 1) : (bagOffsets,)
    > maxIndices的shape 为 includeLastOffset ? (bagOffsets - 1) : (bagOffsets,)
  - 当mode为mean模式：
    > output的shape 为 includeLastOffset? (bagOffsets \* wordsOffsets - 1, embeddingDim) : (bagOffsets \* wordsOffsets, embeddingDim)
    > offset2bag的shape 为 (0,) 或 (bagIndices,)
    > bagSize的shape 为 includeLastOffset ? (bagOffsets - 1) : (bagOffsets,)
    > maxIndices的shape 为 includeLastOffset ? (bagOffsets - 1) : (bagOffsets,)
  - 当mode为max模式：
    > output的shape 为 includeLastOffset ? (bagOffsets \* wordsOffsets - 1, embeddingDim) : (bagOffsets \* wordsOffsets, embeddingDim)
    > offset2bag的shape 为 (0,) 或 (bagIndices,)
    > bagSize的shape 为 includeLastOffset ? (bagOffsets - 1) : (bagOffsets,)
    > maxIndices的shape 为 includeLastOffset ? (bagOffsets \* wordsOffsets - 1, embeddingDim) : (bagOffsets \* wordsOffsets, embeddingDim)

## aclnnEmbeddingBagGetWorkspaceSize

- **参数说明**：

  - weight(aclTensor*, 计算输入)：词嵌入矩阵，包含所有词的嵌入向量，Device侧的aclTensor，shape支持2维，支持非连续的Tensor，数据格式支持ND。
    - Atlas 推理系列产品/Atlas 训练系列产品：数据类型支持FLOAT、FLOAT16。
    - Atlas A2 训练系列产品/Atlas A3 训练系列产品：数据类型支持FLOAT、FLOAT16、BFLOAT16。
  - indices(aclTensor*, 计算输入)：包含索引的张量，指定要从`weight`中提取哪些词的嵌入向量，Device侧的aclTensor。数据类型支持UINT8、INT8、INT16、INT32、INT64，shape支持1维。支持非连续的Tensor，数据格式支持ND。
  - offsets(aclTensor*, 计算输入): 用于将 indices 分割成多个`bag`的偏移量张量，Device侧的aclTensor。数据类型支持UINT8、INT8、INT16、INT32、INT64。shape支持1维。
  - scaleGradByFreq(bool，计算输入): 用于控制是否根据词频缩放梯度，当scaleGradByFreq为true时，会根据词频对梯度进行缩放，当scaleGradByFreq为false时，则不会。
  - mode(int64_t, 计算输入)：用于控制聚合模式，Host侧的整型。0表示sum聚合模式，1表示mean聚合模式，其他表示max聚合模式。
  - sparse(bool, 计算输入)：用于控制稀疏模式，Host侧的bool类型。当为false时，表示weight非稀疏矩阵；当为true时，表示weight是稀疏矩阵。
  - perSampleWeights(aclTensor*, 计算输入): 指定样本权重，Device侧的aclTensor。shape支持1维，数据类型与weight一致，仅在sum模式下，可以不是nullptr，其他模式必须为nullptr。
    - Atlas 推理系列产品/Atlas 训练系列产品：数据类型支持FLOAT、FLOAT16。
    - Atlas A2 训练系列产品/Atlas A3 训练系列产品：数据类型支持FLOAT、FLOAT16、BFLOAT16。
  - includeLastOffset(bool, 计算输入)：控制是否包含最后的偏移，Host侧的bool类型。当为false时，表示不包含最后的偏移；当为true时，表示包含最后的偏移。
  - paddingIdx(int64_t, 计算输入): 控制不参与计算的indices，Host侧的整型，取值范围是[-n,n-1]，其中n是weigit第一维元素个数。
  - output(aclTensor*, 计算输出)：词嵌入矩阵聚合后的结果，Device侧的aclTensor。数据类型与weight一致，shape支持2维。支持非连续的Tensor，数据格式支持ND。
    - Atlas 推理系列产品/Atlas 训练系列产品：数据类型支持FLOAT、FLOAT16。
    - Atlas A2 训练系列产品/Atlas A3 训练系列产品：数据类型支持FLOAT、FLOAT16、BFLOAT16。
  - offset2bag(aclTensor*, 计算输出)：`bag`的起始偏移，Device侧的aclTensor。数据类型支持INT32、INT64，且offset2bag的数据类型和indices与offsets中精度高的一致，shape支持1维。支持非连续的Tensor，数据格式支持ND。
  - bagSize(aclTensor*, 计算输出): 每个`bag`的大小，Device侧的aclTensor。数据类型支持INT32、INT64，且bagSize的数据类型和indices与offsets中精度高的一致，shape支持1维。支持非连续的Tensor，数据格式支持ND。
  - maxIndices(aclTensor*, 计算输出): 当`mode`为max时，词嵌入向量最大值所在的行，Device侧的aclTensor。数据类型支持INT32、INT64，且maxIndices的数据类型和indices与offsets中精度高的一致。当`mode`为max时，shape支持2维；当`mode`非max时，shape支持1维。支持非连续的Tensor，数据格式支持ND。
  - workspaceSize(uint64_t*, 出参)：返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor**, 出参)：返回op执行器，包含了算子计算流程。

- **返回值**：

  aclnnStatus：返回状态码，具体参见aclnn返回码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001(ACLNN_ERR_PARAM_NULLPTR): 1. 传入的 weight、indices、offsets、output、offset2bag、bagSize、maxIndices是空指针。
  返回161002(ACLNN_ERR_PARAM_INVALID): 1. weight数据类型不在支持范围内,weight维度不是2维。
                                      2. indices数据类型不在支持范围内,indices维度不是1维。
                                      3. offsets数据类型不在支持范围内, offsets维度不是1维。
                                      4. indices和offsets的数据类型都不是INT32或INT64。
                                      5. perSampleWeights在传入非nullptr的情况下，数据类型与weight不一致, perSampleWeights不是1维，perSampleWeights元素数量与indices不相等, 在非sum模式下，perSampleWeights不是nullptr。
                                      6. paddingIdx超出范围。
                                      7. output数据类型与weight不一致,shape与定义不符。
                                      8. offset2bag、bagSize、maxIndices数据类型和shape与推导得到的数据类型和shape不符。
  ```

## aclnnEmbeddingBag

- **参数说明**：

  - workspace(void*, 入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnEmbeddingBagGetWorkspaceSize获取。
  - executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream, 入参)：指定执行任务的AscendCL Stream流。

- **返回值**：

  aclnnStatus：返回状态码，具体参见aclnn返回码。

## 约束与限制

sparse与scaleGradByFreq仅支持输入False。