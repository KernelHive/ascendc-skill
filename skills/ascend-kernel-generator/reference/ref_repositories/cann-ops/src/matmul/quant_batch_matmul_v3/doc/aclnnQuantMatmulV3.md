# aclnnQuantMatmulV3

## 支持的产品型号
- 昇腾910B AI处理器
- 昇腾910_93 AI处理器
- 昇腾310P AI处理器
## 接口原型
每个算子分为两段式接口，必须先调用“aclnnQuantMatmulV3GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnQuantMatmulV3”接口执行计算。

- `aclnnStatus aclnnQuantMatmulV3GetWorkspaceSize(const aclTensor* x1, const aclTensor* x2, const aclTensor* scale, const aclTensor* offset, const aclTensor* bias, bool transposeX1, bool transposeX2, const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`

- `aclnnStatus aclnnQuantMatmulV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- 算子功能：完成量化的矩阵乘计算，最小支持输入维度为2维，最大支持输入维度为6维。相似接口有aclnnMm（仅支持2维Tensor作为输入的矩阵乘）和aclnnBatchMatMul（仅支持三维的矩阵乘，其中第一维是Batch维度）。
- 计算公式：

  - 无bias：
  $$
  out = x1@x2 * scale + offset
  $$

  - bias int32：
  $$
  out = (x1@x2 + bias) * scale + offset
  $$

  - bias bfloat16/float32（此场景无offset）：
  $$
  out = x1@x2 * scale + bias
  $$

## aclnnQuantMatmulV3GetWorkspaceSize

- **参数说明：**

  - x1（aclTensor*，计算输入）：公式中的输入x1，device侧的aclTensor。数据格式支持ND。支持最后两根轴转置情况下的非连续tensor，其他场景的非连续的Tensor不支持。shape支持2~6维，在transposeX1为false情况下各个维度表示：（batch，m，k），在transposeX1为true情况下各个维度表示：（batch，k，m），batch可不存在。
    - 昇腾310P AI处理器：数据类型支持INT8。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持INT8、INT32、INT4。当数据类型为INT32、INT4时，为INT4量化场景，当前仅支持transposeX1为false情况。其中当x1数据类型为INT4时，维度表示：（batch，m，k），要求k为偶数，当x1数据类型为INT32时，每个INT32数据存放8个INT4数据，对应维度表示：（batch，m，k // 8），要求k为8的倍数。
  - x2（aclTensor*，计算输入）：公式中的输入x2，device侧的aclTensor。数据格式支持ND格式和昇腾亲和数据排布格式。支持最后两根轴转置情况下的非连续tensor，其他场景的非连续的Tensor不支持。
    - ND格式下，shape支持2~6维，在transposeX2为false情况下各个维度表示：（batch，k，n），在transposeX2为true情况下各个维度表示：（batch，n，k），batch可不存在，其中k与x1的shape中的k一致。
    - 昇腾亲和数据排布格式下，shape支持4~8维。在transposeX2为true情况下各个维度表示：（batch，k1，n1，n0，k0），batch可不存在，其中k0 = 32，n0 = 16，x1 shape中的k和x2 shape中的k1需要满足以下关系：ceilDiv（k，32） = k1。在transposeX2为false情况下各个维度表示：（batch，n1，k1，k0，n0），batch可不存在，其中k0 = 16，n0 = 32，x1 shape中的k和x2 shape中的k1需要满足以下关系：ceilDiv（k，16） = k1。
    可使用aclnnCalculateMatmulWeightSizeV2接口以及aclnnTransMatmulWeight接口完成输入Format从ND到昇腾亲和数据排布格式的转换。
    - 昇腾310P AI处理器：数据类型支持INT8。当输入x2为昇腾亲和数据排布格式时，不支持transposeX2为false的场景。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持INT8、INT32、INT4。当数据类型为INT32、INT4时，为INT4量化场景，当前仅支持2维ND格式。
      - 数据类型为INT4时，在transposeX2为true情况下各个维度表示：（n，k），要求k为偶数；在transposeX2为false情况下各个维度表示：（k，n），要求n为偶数。
      - 数据类型为INT32时，每个INT32数据存放8个INT4数据，在transposeX2为true情况下各个维度表示：（n，k // 8），要求k为8的倍数；在transposeX2为false情况下各个维度表示：（k，n // 8），要求n为8的倍数。
      - 可使用aclnnConvertWeightToINT4Pack接口完成x2从INT32（1个int32在0~3bit位存储1个int4）到INT32（1个int32存储8个int4）或INT4（1个int4表示1个int4）的数据格式转换，具体参见aclnnConvertWeightToINT4Pack接口。

  - scale（aclTensor*，计算输入）：表示量化参数，公式中的输入scale，device侧的aclTensor。数据格式支持ND。shape是1维（t，），t = 1或n，其中n与x2的n一致。

    - 昇腾310P AI处理器：数据类型支持UINT64、INT64。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持UINT64、INT64、FLOAT32、BFLOAT16。
    - 当原始输入类型不满足约束与限制中的数据类型组合时，需要提前调用TransQuantParamV2算子的aclnn接口来将scale转成INT64、UINT64数据类型。

  - offset（aclTensor*，计算输入）：公式中的输入offset，device侧的aclTensor。数据类型支持FLOAT32，数据格式支持ND，shape是1维（t，），t = 1或n，其中n与x2的n一致。

  - bias（aclTensor*，计算输入）：公式中的输入bias，device侧的aclTensor。可选参数，数据格式支持ND。shape支持1维（n，）或3维（batch，1，n），n与x2的n一致。当out的shape为2、4、5、6维时，bias的shape只支持1维（n，）。
    - 昇腾310P AI处理器：数据类型支持INT32。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持INT32、BFLOAT16、FLOAT32。当x1和x2为int32、int4时，bias的shape只支持1维（n，）。

  - transposeX1（bool，计算输入）：表示x1的输入shape是否包含transpose。在transposeX1为false情况下各个维度表示：（batch，m，k），在transposeX1为true情况下各个维度表示：（batch，k，m），batch可不存在。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：x1和x2为int32、int4时，transposeX1仅支持false。

  - transposeX2（bool，计算输入）：表示x2的输入shape是否包含transpose。
    - ND格式下，在transposeX2为false情况下各个维度表示：（batch，k，n），在transposeX2为true情况下各个维度表示：（batch，n，k），batch可不存在，其中k与x1的shape中的k一致。
    - 昇腾亲和数据排布格式下，在transposeX2为true情况下各个维度表示：（batch，k1，n1，n0，k0），batch可不存在，其中k0 = 32，n0 = 16，x1 shape中的k和x2 shape中的k1需要满足以下关系：ceilDiv（k，32） = k1。在transposeX2为false情况下各个维度表示：（batch，n1，k1，k0，n0），batch可不存在，其中k0 = 16，n0 = 32，x1 shape中的k和x2 shape中的k1需要满足以下关系：ceilDiv（k，16） = k1。

  - out（aclTensor*，计算输出）：公式中的输出out，device侧的aclTensor。数据格式支持ND。支持非连续的Tensor，shape支持2~6维，（batch，m，n），batch可不存在，支持x1与x2的batch维度broadcast，输出batch与broadcast之后的batch一致，m与x1的m一致，n与x2的n一致。
    - 昇腾310P AI处理器：数据类型支持FLOAT16、INT8。
    - 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持FLOAT16、INT8、BFLOAT16、INT32。

  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。

  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见aclnn返回码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  - 161001(ACLNN_ERR_PARAM_NULLPTR)：
  1. 传入的x1、x2、scale或out是空指针。
  - 161002(ACLNN_ERR_PARAM_INVALID)：
  1. x1、x2、bias、scale、offset或out的数据类型和数据格式不在支持的范围之内。
  2. x1、x2、bias、scale、offset或out的shape不满足校验条件。
  3. x1、x2、bias、scale、offset或out是空tensor。
  4. x1与x2的最后一维大小超过65535，x1的最后一维指transposeX1为true时的m或transposeX1为false时的k，x2的最后一维指transposeX2为true时的k或transposeX2为false时的n。
  ```

## aclnnQuantMatmulV3

- **参数说明：**
- workspace(void*, 入参)：在Device侧申请的workspace内存地址。
- workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnQuantMatmulV3GetWorkspaceSize获取。
- executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。
- stream(aclrtStream, 入参)：指定执行任务的AscendCL Stream流。

- **返回值：**

aclnnStatus：返回状态码，具体参见aclnn返回码。

## 约束与限制
输入和输出支持以下数据类型组合：
- 昇腾310P AI处理器：

| x1 | x2 | scale | offset | bias | out |
| ------- | ------- | ------ | ------ | ------- | ------- |
| int8 | int8 | uint64/int64 | null | null/int32  |  float16	|
| int8 | int8 | uint64/int64 | null/float32 | null/int32  |  int8	|

- 昇腾910B AI处理器、昇腾910_93 AI处理器：

| x1 | x2 | scale | offset | bias | out |
| ------- | ------- | ------ | ------ | ------- | ------- |
| int8 | int8 | uint64/int64 | null | null/int32  |  float16	|
| int8 | int8 | uint64/int64 | null/float32 | null/int32  |  int8	|
| int8 | int8 | float32/bfloat16 | null | null/int32/bfloat16/float32  |  bfloat16 |
| int4/int32 | int4/int32 | uint64/int64 | null | null/int32  |  float16 |
| int8 | int8 | float32/bfloat16 | null | null/int32  | int32 |

## 调用示例

详见[QuantBatchMatmulV3自定义算子样例说明算子调用章节](../README.md#算子调用)


