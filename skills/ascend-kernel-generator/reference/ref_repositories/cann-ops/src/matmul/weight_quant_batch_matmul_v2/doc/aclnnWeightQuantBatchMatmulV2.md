# aclnnWeightQuantBatchMatmulV2

## 支持的产品型号

- 昇腾910B AI处理器
- 昇腾910_93 AI处理器
- 昇腾310P AI处理器

## 接口原型

每个算子分为两段式接口，必须先调用“aclnnWeightQuantBatchMatmulV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnWeightQuantBatchMatmulV2”接口执行计算。
  - `aclnnStatus aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *antiquantScale, const aclTensor *antiquantOffsetOptional, const aclTensor *quantScaleOptional, const aclTensor *quantOffsetOptional, const aclTensor *biasOptional, int antiquantGroupSize, const aclTensor *y, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnWeightQuantBatchMatmulV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- **算子功能**：完成一个输入为伪量化场景的矩阵乘计算，并可以实现对于输出的量化计算。
- **计算公式**：

  $$
  y = x @ ANTIQUANT(weight) + bias
  $$

  公式中的$weight$为伪量化场景的输入，其反量化公式$ANTIQUANT(weight)$为

  $$
  ANTIQUANT(weight) = (weight + antiquantOffset) * antiquantScale
  $$

  当客户配置quantScaleOptional输入时，会对输出进行量化处理，其量化公式为

  $$
  \begin{aligned}
  y &= QUANT(x @ ANTIQUANT(weight) + bias) \\
  &= (x @ ANTIQUANT(weight) + bias) * quantScale + quantOffset \\
  \end{aligned}
  $$

  当客户配置quantScaleOptional输入为nullptr, 则直接输出:

  $$
  y = x @ ANTIQUANT(weight) + bias
  $$

## aclnnWeightQuantBatchMatmulV2GetWorkspaceSize

- **参数说明**

  - x(aclTensor*, 计算输入)：公式中的输入`x`，数据格式支持ND。非连续的Tensor仅支持transpose场景。维度支持2维，shape支持(m, k)，其中Reduce维度k需要与`weight`的Reduce维度k大小相等。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持FLOAT16、BFLOAT16。
    - 昇腾310P AI处理器：数据类型支持FLOAT16。
  - weight（aclTensor*， 计算输入）：公式中的输入`weight`，数据格式支持ND、FRACTAL_NZ。维度支持2维，Reduce维度k需要与`x`的Reduce维度k大小相等。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持INT8、INT4、INT32（当weight数据格式为FRACTAL_NZ且数据类型为INT4/INT32时，或者当weight数据格式为ND且数据类型为INT32时，仅在INT4Pack场景支持，需配合aclnnConvertWeightToINT4Pack接口完成从INT32到INT4Pack的转换，以及从ND到FRACTAL_NZ的转换，若数据类型为INT4，则weight的内轴应为偶数。非连续的Tensor仅支持transpose场景。shape支持(k, n)。

      对于不同伪量化算法模式，weight的数据格式为FRACTAL_NZ仅在如下场景下支持:
      - per_channel模式：
        - weight的数据类型为INT8，y的数据类型为非INT8。
        - weight的数据类型为INT4/INT32，weight转置，y的数据类型为非INT8。
      - per_group模式：weight的数据类型为INT4/INT32，weight非转置，x非转置，antiquantGroupSize为64或128，k为antiquantGroupSize对齐，n为64对齐，y的数据类型为非INT8。
    - 昇腾310P AI处理器：数据类型支持INT8。只支持per_channel场景。输入shape需要为（n, k）。数据格式为FRACTAL_NZ时，配合aclnnCalculateMatmulWeightSizeV2以及aclnnTransMatmulWeight完成输入Format从ND到FRACTAL_NZ的转换。
  - antiquantScale(aclTensor*, 计算输入)：实现输入反量化计算的反量化scale参数，反量化公式中的输入`antiquantScale`。数据格式支持ND。

    - 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持FLOAT16、BFLOAT16、UINT64、INT64（当FLOAT16、BFLOAT16时，数据类型要求和输入`x`保持一致；当为UINT64、INT64时，x仅支持FLOAT16，不转置，weight仅支持int8，ND转置，模式仅支持per_channel，quantScaleOptional和quantOffsetOptional必须传入空指针，m仅支持[1, 96]，k和n要求64对齐，需要首先配合aclnnCast接口完成FLOAT16到FLOAT32的转换，详情可参考样例，再配合aclnnTransQuantParamV2接口完成FLOAT32到UINT64的转换。非连续的Tensor仅支持transpose场景。

      对于不同伪量化算法模式，antiquantScale支持的shape如下:
      - per_tensor模式：输入shape为(1,)或(1, 1)。
      - per_channel模式：输入shape为(1, n)或(n,)。
      - per_group模式：输入shape为(ceil(k, group_size), n)。
    - 昇腾310P AI处理器：数据类型支持FLOAT16，数据类型要求和输入`x`保持一致。

      对于不同伪量化算法模式，antiquantScale支持的shape如下:
      - per_tensor模式：输入shape为(1,)或(1, 1)。
      - per_channel模式：输入shape为(n, 1)或(n,)，不支持非连续的Tensor。
  - antiquantOffsetOptional（aclTensor*, 计算输入）：实现输入反量化计算的反量化offset参数，反量化公式中的输入`antiquantOffset`。可选输入, 当不需要时为空指针；存在时shape要求与`antiquantScale`一致。数据格式支持ND。

    - 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持FLOAT16、BFLOAT16、INT32，（数据类型为FLOAT16、BFLOAT16时，数据类型要求和输入`x`的数据类型保持一致；数据类型为INT32类型时，数据范围限制为[-128, 127]，x仅支持FLOAT16，weight仅支持int8，antiquantScale仅支持UINT64/INT64）。非连续的Tensor仅支持transpose场景。
    - 昇腾310P AI处理器：数据类型支持FLOAT16，数据类型要求和输入`x`保持一致。
  - quantScaleOptional（aclTensor*, 计算输入）：实现输出量化计算的量化参数，由量化公式中的quantScale和quantOffset的数据通过`aclnnTransQuantParam`接口转化得到。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持UINT64，数据格式支持ND。不支持非连续的Tensor。可选输入，当不需要时为空指针；对于不同的伪量化算法模式，支持的shape如下:
      - per_tensor模式：输入shape为(1,)或(1, 1)。
      - per_channel模式：输入shape为(1, n)或(n,)。
    - 昇腾310P AI处理器：预留参数，暂未使用，固定传入空指针。
  - quantOffsetOptional（aclTensor*, 计算输入）：实现输出量化计算的量化offset参数，量化公式中的输入`quantOffset`，
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持FLOAT，数据格式支持ND。可选输入, 当不需要时为空指针；存在时shape要求与`quantScaleOptional`一致。不支持非连续的Tensor。
    - 昇腾310P AI处理器：预留参数，暂未使用，固定传入空指针。
  - biasOptional（aclTensor\*, 计算输入）：偏置输入，公式中的输入`bias`。可选输入, 当不需要时为空指针；存在输入时支持1维或2维，shape支持(n,)或(1, n)。数据格式支持ND。不支持非连续的Tensor。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持FLOAT16、FLOAT。当`x`数据类型为BFLOAT16时，本参数要求为FLOAT；当`x`数据类型为FLOAT16时，本参数要求为FLOAT16。
    - 昇腾310P AI处理器：数据类型支持FLOAT16。
  - antiquantGroupSize（int, 计算输入）：表示伪量化per_group算法模式下，对输入`weight`进行反量化计算的groupSize输入，描述一组反量化参数对应的待反量化数据量在Reduce方向的大小。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：当伪量化算法模式不为per_group时传入0；当伪量化算法模式为per_group时传入值的范围为[32, k-1]且值要求是32的倍数。
    - 昇腾310P AI处理器：不支持per_group算法模式，固定传入0。

  - y（aclTensor\*, 计算输出）：计算输出，公式中的`y`。维度支持2维，shape支持(m, n)。数据格式支持ND。不支持非连续的Tensor。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持FLOAT16、BFLOAT16、INT8。当`quantScaleOptional`存在时，数据类型为INT8；当`quantScaleOptional`不存在时，数据类型支持FLOAT16、BFLOAT16，且与输入`x`的数据类型一致。
    - 昇腾310P AI处理器：数据类型支持FLOAT16。
  - workspaceSize（uint64_t\*, 出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\**, 出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见aclnn返回码。

```
第一段接口完成入参校验，出现以下场景时报错：
161001 (ACLNN_ERR_PARAM_NULLPTR)：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
161002 (ACLNN_ERR_PARAM_INVALID)：
- 传入x、weight、antiquantScale、antiquantOffsetOptional、quantScaleOptional、quantOffsetOptional、biasOptional、y的shape维度不符合要求。
- 传入x、weight、antiquantScale、antiquantOffsetOptional、quantScaleOptional、quantOffsetOptional、biasOptional、y的数据类型不在支持的范围之内。
- x、weight的reduce维度(k)不相等。
- antiquantOffsetOptional存在输入时，shape与antiquantScale不相同。
- quantOffsetOptional存在输入时，shape与quantScale不相同。
- biasOptional的shape不符合要求。
- antiquantGroupSize值不符合要求。
- quantOffsetOptional存在时，quantScaleOptional是空指针。
- 输入的k、n值不在[1, 65535]范围内；
- x矩阵为非转置时，m不在[1, 2^31-1]范围内；转置时，m不在[1, 65535]范围内
- 不支持空tensor场景。
- 输入tensor的数据格式不在支持范围内。
- 传入x、weight、antiquantScale、antiquantOffsetOptional、quantScaleOptional、quantOffsetOptional、biasOptional、y的连续性不符合要求。
361001(ACLNN_ERR_RUNTIME_ERROR): 产品型号不支持。
```
## aclnnWeightQuantBatchMatmulV2

- **参数说明**

 - workspace(void\*, 入参)：在Device侧申请的workspace内存地址。
 - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnWeightQuantBatchMatmulV2GetWorkspaceSize获取。
 - executor(aclOpExecutor\*, 入参)：op执行器，包含了算子计算流程。
 - stream(aclrtStream, 入参)：指定执行任务的 AscendCL Stream流。

- **返回值：**

 aclnnStatus：返回状态码，具体参见aclnn返回码。


## 约束与限制

per_channel模式：为提高性能，推荐使用transpose后的weight输入。m范围为[65, 96]时，推荐使用数据类型为UINT64/INT64的antiquantScale。

详见[WeightQuantBatchMatmulV2自定义算子样例说明算子调用章节](../README.md#算子调用)



