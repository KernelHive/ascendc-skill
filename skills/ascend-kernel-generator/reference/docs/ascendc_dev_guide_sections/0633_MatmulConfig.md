###### MatmulConfig

`MatmulConfig` 模板参数用于配置 Matmul 模板信息以及相关的配置参数。不配置默认使能 Norm 模板，Norm 模板的介绍请参考表“模板特性”。`MatmulConfig` 的参数说明见表 15-593。

`MatmulConfig` 的定义方式有：

- 该模板参数可选取提供的模板之一，当前提供的 `MatmulConfig` 模板取值范围为 `[CFG_NORM、CFG_MDL、CFG_IBSHARE_NORM、MM_CFG_BB]`，分别对应默认的 Norm、MDL、IBShare、BasicBlock 模板。各模板的介绍请参考表 15-592。
- 该模板参数可以基于各类获取模板的接口，自定义模板参数配置，获取自定义模板。各类获取模板的接口包括：`GetNormalConfig`、`GetMDLConfig`、`GetSpecialMDLConfig`、`GetIBShareNormConfig`、`GetBasicConfig`、`GetSpecialBasicConfig`。
- 另外，`MatmulConfig` 可拆分为 `MatmulShapeParams`、`MatmulQuantParams`、`MatmulBatchParams`、`MatmulFuncParams` 二级子 Config，使用 `GetMMConfig` 接口，设置需要的二级子 Config 和 `MatmulConfigMode`，可以更加灵活的获取自定义的模板参数配置 `MatmulConfig`。

## 模板特性

| 模板 | 实现 | 优点 | 推荐使用场景 |
|------|------|------|--------------|
| Norm | 支持 L1 缓存多个基本块，MTE2 分多次从 GM 搬运基本块到 L1，每次搬运一份基本块，已搬的基本块不清空。（举例说明：Tiling 结构体中的 `depthA1=6`，代表搬入 6 份 A 矩阵基本块到 L1，1 次搬运一份基本块，MTE2 进行 6 次搬运）。 | 可以提前启动 MTE1 流水（因为搬 1 份基本块就可以做 MTE1 后面的运算）。 | 默认使能 Norm 模板。 |
| MDL, Special MDL | 支持 L1 缓存多个基本块，MTE2 从 GM 到 L1 的搬运为一次性“大包”搬运。（举例说明：Tiling 结构体中的 `depthA1=6`，代表一次性搬入 6 份 A 矩阵基本块到 L1，MTE2 进行 1 次搬运）。MDL 模板与 SpecialMDL 模板的差异见表 15-593。 | 对于一般的大 shape 场景，可以减少 MTE2 的循环搬运，提升性能。 | 大 shape 场景。 |
| IBShare | MIX 场景下，A 矩阵或 B 矩阵 GM 地址相同的时候，通过共享 L1 Buffer，减少 MTE2 搬运。 | 减少 MTE2 搬运，提升性能。 | MIX 场景多个 AIV 的 A 矩阵或 B 矩阵 GM 地址相同。<br>注意：IBShare 模板要求多个 AIV 复用的 A/B 矩阵必须在 L1 Buffer 上全载。 |
| BasicBlock | 在无尾块的场景，基本块大小确定的情况下，通过 `GetBasicConfig` 接口配置输入的基本块大小，固定 MTE1 每次搬运的矩阵大小及每次矩阵乘计算的矩阵大小，减少参数计算量。 | 减少 MTE1 矩阵搬运和矩阵乘计算过程中的参数计算开销。 | 无尾块，基本块（baseM, baseN）大小确定。 |

## MatmulConfig 参数说明

### doNorm

- **说明**：使能 Norm 模板。参数取值如下：
  - `true`：使能 Norm 模板。
  - `false`：不使能 Norm 模板。
  不指定模板的情况默认使能 Norm 模板。
- **支持模板**：Norm

### doBasicBlock

- **说明**：使能 BasicBlock 模板。模板参数取值如下：
  - `true`：使能 BasicBlock 模板。
  - `false`：不使能 BasicBlock 模板。
  调用 `GetBasicConfig` 接口获取 BasicBlock 模板时，该参数被置为 `true`。
- **注意**：
  - BasicBlock 模板暂不支持输入为 `int8_t`、`int4_t` 数据类型的 A、B 矩阵，支持 `half`/`float`/`bfloat16_t` 数据类型的 A、B 矩阵。
  - BasicBlock 模板暂不支持 A 矩阵为标量数据 Scalar 或向量数据 Vector。
  - BasicBlock 模板暂不支持 `ScheduleType::OUTER_PRODUCT` 的数据搬运模式。
  - Atlas A2 训练系列产品/Atlas A2 推理系列产品支持该参数。
  - Atlas A3 训练系列产品/Atlas A3 推理系列产品支持该参数。
  - Atlas 推理系列产品 AI Core 不支持设置为 `true`。
  - Atlas 200I/500 A2 推理产品不支持设置为 `true`。
- **支持模板**：BasicBlock

### doMultiDataLoad

- **说明**：使能 MDL 模板。参数取值如下：
  - `true`：使能 MDL 模板。
  - `false`：不使能 MDL 模板。
- **支持模板**：MDL

### basicM

- **说明**：与 `TCubeTiling` 结构体中的 `baseM` 参数含义相同，Matmul 计算时 base 块 M 轴长度，以元素为单位。
- **支持模板**：BasicBlock

### basicN

- **说明**：与 `TCubeTiling` 结构体中的 `baseN` 参数含义相同，Matmul 计算时 base 块 N 轴长度，以元素为单位。
- **支持模板**：BasicBlock

### basicK

- **说明**：与 `TCubeTiling` 结构体中的 `baseK` 参数含义相同，Matmul 计算时 base 块 K 轴长度，以元素为单位。
- **支持模板**：BasicBlock

### intrinsicsCheck

- **说明**：当左矩阵或右矩阵在单核上内轴（即尾轴）大于等于 65535（元素个数）时，是否使能循环执行数据从 Global Memory 到 L1 Buffer 的搬入。例如，左矩阵 A[M, K]，单核上的内轴数据 `singleCoreK` 大于 65535，配置该参数为 `true` 后，API 内部通过循环执行数据的搬入。参数取值如下：
  - `false`：当左矩阵或右矩阵在单核上内轴大于等于 65535 时，不使能循环执行数据的搬入（默认值）。
  - `true`：当左矩阵或右矩阵在单核上内轴大于等于 65535 时，使能循环执行数据的搬入。
- **支持模板**：所有模板

### isNBatch

- **说明**：是否多 Batch 输入多 Batch 输出。仅对 BatchMatmul 有效，使能该参数后，仅支持 Norm 模板，且需调用 `IterateNBatch` 实现多 Batch 输入多 Batch 输出。参数取值如下：
  - `false`：不使能多 Batch（默认值）。
  - `true`：使能多 Batch。
- **支持模板**：Norm

### enVecND2NZ

- **说明**：使能通过 vector 指令进行 ND2NZ。使能时需要设置 `SetLocalWorkspace`。参数取值如下：
  - `false`：不使能通过 vector 指令进行 ND2NZ（默认值）。
  - `true`：使能通过 vector 指令进行 ND2NZ。
  针对 Atlas 推理系列产品 AI Core，在 Unified Buffer 空间足够的条件下（Unified Buffer 空间大于 2 倍 `TCubeTiling` 的 `transLength` 参数），建议优先使能该参数，搬运性能更好。
- **支持模板**：所有模板

### doSpecialBasicBlock

- **说明**：使能 SpecialBasicBlock 模板。参数取值如下：
  - `true`：使能 SpecialBasicBlock 模板。
  - `false`：不使能 SpecialBasicBlock 模板。
  本质上也是 BasicBlock 模板，但消除了头开销 scalar 计算。
- **支持模板**：预留参数

### doMTE2Preload

- **说明**：在 MTE2 流水间隙较大，且 M/N 数值较大时可通过该参数开启对应 M/N 方向的预加载功能，开启后能减小 MTE2 间隙，提升性能。预加载功能仅在 MDL 模板有效（不支持 SpecialMDL 模板）。参数取值如下：
  - `0`：不开启（默认值）。
  - `1`：开启 M 方向 preload。
  - `2`：开启 N 方向 preload。
- **注意**：开启 M/N 方向的预加载功能时需保证 K 全载且 M/N 方向开启 double buffer；其中，M 方向的 K 全载条件为：`singleCoreK/baseK <= stepKa`；N 方向的 K 全载条件为：`singleCoreK/baseK <= stepKb`。
  该参数的使用样例请参考 M/N 方向预加载 Matmul 算子样例。
- **支持模板**：MDL

### singleCoreM

- **说明**：单核内 M 轴 shape 大小，以元素为单位。
- **支持模板**：预留参数

### singleCoreN

- **说明**：单核内 N 轴 shape 大小，以元素为单位。
- **支持模板**：预留参数

### singleCoreK

- **说明**：单核内 K 轴 shape 大小，以元素为单位。
- **支持模板**：预留参数

### stepM

- **说明**：左矩阵在 A1 中缓存的 bufferM 方向上 baseM 的倍数。
- **支持模板**：预留参数

### stepN

- **说明**：右矩阵在 B1 中缓存的 bufferN 方向上 baseN 的倍数。
- **支持模板**：预留参数

### baseMN

- **说明**：`baseM * baseN` 的大小。
- **支持模板**：预留参数

### singleCoreMN

- **说明**：`singleCoreM * singleCoreN` 的大小。
- **支持模板**：预留参数

### enUnitFlag

- **说明**：使能 UnitFlag 功能，使计算与搬运流水并行，提高性能。Norm、IBShare 下默认使能，MDL 下默认不使能。参数取值如下：
  - `false`：不使能 UnitFlag 功能。
  - `true`：使能 UnitFlag 功能。
  该参数的使用样例请参考 `matmul_unitflag` 算子样例。
- **支持模板**：MDL、Norm、IBShare

### isPerTensor

- **说明**：A 矩阵 half 类型输入且 B 矩阵 `int8_t` 类型输入场景，使能 B 矩阵量化时是否为 per tensor。
  - `true`：per tensor 量化。
  - `false`：per channel 量化。
- **支持模板**：MDL、SpecialMDL

### hasAntiQuantOffset

- **说明**：A 矩阵 half 类型输入且 B 矩阵 `int8_t` 类型输入场景，使能 B 矩阵量化时是否使用 offset 系数。
- **支持模板**：MDL、SpecialMDL

### doIBShareNorm

- **说明**：使能 IBShare 模板。参数取值如下：
  - `false`：不使能 IBShare。
  - `true`：使能 IBShare。
  IBShare 的功能是能够复用 L1 上相同的 A 矩阵或 B 矩阵数据，开启后在数据复用场景能够避免重复搬运数据到 L1。
- **支持模板**：IBShare

### doSpecialMDL

- **说明**：使能 SpecialMDL 模板。参数取值如下：
  - `true`：使能 SpecialMDL 模板。
  - `false`：不使能 SpecialMDL 模板。
  MDL 模板的一种特殊场景：Matmul K 方向不全载时（`singleCoreK/baseK > stepKb`），默认仅支持 `stepN` 设置为 1，使能 SpecailMDL 模板后支持 `stepN=2`。
- **注意**：使能 SpecialMDL 模板时，`doMultiDataLoad` 参数取值必须为 `false`。
- **支持模板**：SpecialMDL

### enableInit

- **说明**：是否启用 Init 函数，不使能 Init 函数能够提升常量传播效果，优化性能。默认使能。
  - `false`：不使能 Init 函数。
  - `true`：使能 Init 函数。
- **支持模板**：所有模板

### batchMode

- **说明**：BatchMatmul 场景中 Layout 类型为 NORMAL 时，设置 BatchMatmul 输入 A/B 矩阵的多 batch 数据总和与 L1 Buffer 的大小关系。参数取值如下：
  - `BatchMode::BATCH_LESS_THAN_L1`：多 batch 数据总和 < L1 Buffer Size；
  - `BatchMode::BATCH_LARGE_THAN_L1`：多 batch 数据总和 > L1 Buffer Size；
  - `BatchMode::SINGLE_LARGE_THAN_L1`：单 batch 数据总和 > L1 Buffer Size。
- **支持模板**：Norm

### enableEnd

- **说明**：Matmul 计算过程中是否需要调用 End 函数，该参数可用于优化性能。参数取值如下：
  - `true`：Matmul 计算过程中需要调用 End 函数（默认值）。
  - `false`：不需要调用 End 函数。End 处理相关的代码都会在编译期删除，从而优化性能。例如，异步场景不需要调用 End 函数，可以将该参数置为 `false`。
- **支持模板**：所有模板

### enableGetTensorC

- **说明**：Matmul 计算过程中是否需要调用 `GetTensorC` 函数，该参数可用于优化性能。参数取值如下：
  - `true`：Matmul 计算过程中需要调用 `GetTensorC` 函数（默认值）。
  - `false`：不需要调用 `GetTensorC` 函数。`GetTensorC` 处理相关的代码都会在编译期删除，从而优化性能。
- **支持模板**：所有模板

### enableSetOrgShape

- **说明**：Matmul 计算过程中是否需要调用 `SetOrgShape` 函数，该参数可用于优化性能。参数取值如下：
  - `true`：Matmul 计算过程中需要调用 `SetOrgShape` 函数（默认值）。
  - `false`：不需要调用 `SetOrgShape` 函数。`SetOrgShape` 处理相关的代码都会在编译期删除，从而优化性能。
- **支持模板**：所有模板

### enableSetBias

- **说明**：是否使能计算 Bias。该参数可用于优化性能。参数取值如下：
  - `true`：使能计算 Bias（默认值）。若输入带有 Bias，实现过程中做 Bias 的搬运、计算等。
  - `false`：不计算 Bias。Bias 处理相关的代码都会在编译期删除，从而优化性能。
- **支持模板**：MDL

### enableSetTail

- **说明**：Matmul 计算过程中是否需要调用 `SetTail` 函数，该参数可用于优化性能。参数取值如下：
  - `true`：Matmul 计算过程中需要调用 `SetTail` 函数（默认值）。
  - `false`：不需要调用 `SetTail` 函数。`SetTail` 处理相关的代码都会在编译期删除，从而优化性能。
- **支持模板**：所有模板

### enableQuantVector

- **说明**：Matmul 计算过程中是否需要调用 `SetQuantVector` 和 `SetQuantScalar` 函数，该参数可用于优化性能。参数取值如下：
  - `true`：Matmul 计算过程中需要调用 `SetQuantVector` 和 `SetQuantScalar` 函数（默认值）。
  - `false`：不需要调用 `SetQuantVector` 和 `SetQuantScalar` 函数。`SetQuantVector` 和 `SetQuantScalar` 处理相关的代码都会在编译期删除，从而优化性能。
- **支持模板**：所有模板

### enableSetDefineData

- **说明**：使能模板参数 `MatmulCallBack`（自定义回调函数）时，用于允许/禁止设置回调函数需要的计算数据或在 GM 上存储的数据地址等信息。参数取值如下：
  - `true`：允许设置（默认值）。
  - `false`：不允许设置。`SetSelfDefineData` 处理相关的代码都会在编译期删除，从而优化性能。
- **支持模板**：MDL

### iterateMode

- **说明**：用于优化 Matmul 计算的头开销。具体为，对 Iterate 系列接口（包括 `Iterate`、`IterateAll`、`IterateBatch`、`IterateNBatch`）的优化，当使能某种模式时，表示 Matmul 计算过程中只调用该种模式对应的一个 Iterate 系列接口，其它 Iterate 系列接口相关的代码都会在编译期删除，从而优化性能。该参数为 `IterateMode` 类型。参数取值如下：
  - `ITERATE_MODE_NORMAL`：对于 Iterate 系列接口，Matmul 计算过程中只调用 `Iterate` 接口。
  - `ITERATE_MODE_ALL`：对于 Iterate 系列接口，Matmul 计算过程中只调用 `IterateAll` 接口。
  - `ITERATE_MODE_BATCH`：对于 Iterate 系列接口，Matmul 计算过程中只调用 `IterateBatch` 接口。
  - `ITERATE_MODE_N_BATCH`：对于 Iterate 系列接口，Matmul 计算过程中只调用 `IterateNBatch` 接口。
  - `ITERATE_MODE_DEFAULT`：默认值，不限定调用 Iterate 系列接口的个数，不使能计算头开销的优化。
- **注意**：
  - Atlas A3 训练系列产品/Atlas A3 推理系列产品支持该参数。
  - Atlas A2 训练系列产品/Atlas A2 推理系列产品支持该参数。
  - Atlas 推理系列产品 AI Core 不支持该参数。
  - Atlas 200I/500 A2 推理产品不支持该参数。
- **支持模板**：所有模板

### enableReuse

- **说明**：`SetSelfDefineData` 函数设置的回调函数中的 `dataPtr` 是否直接传递计算数据。若未调用 `SetSelfDefineData` 设置 `dataPtr`，该参数仅支持默认值 `true`。参数取值如下：
  - `true`：直接传递计算数据，仅限单个值。
  - `false`：传递 GM 上存储的数据地址信息。
- **支持模板**：Norm、MDL

### enableUBReuse

- **说明**：是否使能 Unified Buffer 复用。在 Unified Buffer 空间足够的条件下（Unified Buffer 空间大于 4 倍 `TCubeTiling` 的 `transLength` 参数），使能该参数后，Unified Buffer 空间分为互不重叠的两份，分别存储 Matmul 计算相邻前后两轮迭代的数据，后一轮迭代数据的搬入将不必等待前一轮迭代的 Unified Buffer 空间释放，从而优化流水。参数取值如下：
  - `true`：使能 Unified Buffer 复用。
  - `false`：不使能 Unified Buffer 复用。
- **注意**：
  - Atlas A3 训练系列产品/Atlas A3 推理系列产品不支持该参数。
  - Atlas A2 训练系列产品/Atlas A2 推理系列产品不支持该参数。
  - Atlas 推理系列产品 AI Core 支持该参数。
  - Atlas 200I/500 A2 推理产品不支持该参数。
- **支持模板**：MDL

### enableL1CacheUB

- **说明**：是否使能 L1 Buffer 缓存 Unified Buffer 计算块。建议在 MTE3 和 MTE2 流水串行较多的场景使用。参数取值如下：
  - `true`：使能 L1 Buffer 缓存 Unified Buffer 计算块。
  - `false`：不使能 L1 Buffer 缓存 Unified Buffer 计算块。
  若要使能 L1 Buffer 缓存 Unified Buffer 计算块，必须在 Tiling 实现中调用 `SetMatmulConfigParams` 接口将参数 `enableL1CacheUBIn` 设置为 `true`。
- **注意**：
  - Atlas A3 训练系列产品/Atlas A3 推理系列产品不支持该参数。
  - Atlas A2 训练系列产品/Atlas A2 推理系列产品不支持该参数。
  - Atlas 推理系列产品 AI Core 支持该参数。
  - Atlas 200I/500 A2 推理产品不支持该参数。
- **支持模板**：MDL

### intraBlockPartSum

- **说明**：用于分离模式下的 Vector、Cube 计算融合场景，使能两个 AIV 核的一次计算结果（`baseM * baseN` 大小的矩阵分片）在 L0C Buffer 上累加，参数取值如下：
  - `false`：不使能两个 AIV 核的计算结果在 L0C Buffer 上的累加（默认值）。
  - `true`：使能两个 AIV 核的计算结果在 L0C Buffer 上的累加。
- **注意**：
  - Atlas A3 训练系列产品/Atlas A3 推理系列产品支持该参数。
  - Atlas A2 训练系列产品/Atlas A2 推理系列产品支持该参数。
  - Atlas 推理系列产品 AI Core 不支持该参数。
  - Atlas 200I/500 A2 推理产品不支持该参数。
- **支持模板**：Norm

### IterateOrder

- **说明**：Matmul 做矩阵运算的循环迭代顺序，与表 15-642 中的 `iterateOrder` 参数含义相同。当 `ScheduleType` 参数取值为 `ScheduleType::OUTER_PRODUCT` 时，本参数生效。参数取值如下：
  ```cpp
  enum class IterateOrder {
      ORDER_M = 0, // 先往 M 轴方向偏移再往 N 轴方向偏移
      ORDER_N,     // 先往 N 轴方向偏移再往 M 轴方向偏移
      UNDEF,       // 当前无效
  };
  ```
- **注**：Norm 模板的 Matmul 场景、MDL 模板使用时，若 `IterateOrder` 取值 `ORDER_M`，`TCubeTiling` 结构中的 `stepN` 需要大于 1，`IterateOrder` 取值 `ORDER_N` 时，`TCubeTiling` 结构中的 `stepM` 需要大于 1。
  该参数的使用样例请参考 M/N 方向流水并行 Matmul 算子样例。
- **注意**：
  - Atlas A3 训练系列产品/Atlas A3 推理系列产品支持该参数。
  - Atlas A2 训练系列产品/Atlas A2 推理系列产品支持该参数。
  - Atlas 推理系列产品 AI Core 不支持该参数。
  - Atlas 200I/500 A2 推理产品不支持该参数。
- **支持模板**：Norm、MDL

### scheduleType

- **说明**：配置 Matmul 数据搬运模式。参数取值如下：
  - `ScheduleType::INNER_PRODUCT`：默认模式，在 K 方向上做 MTE1 的循环搬运；
  - `ScheduleType::OUTER_PRODUCT`：在 M 或 N 方向上做 MTE1 的循环搬运；使能后，需要与 `IterateOrder` 参数配合使用。
- **该配置当前只在 BatchMatmul 场景（使能 Norm 模板）或 Matmul 场景（使能 MDL 模板或 Norm 模板）生效**：
  - 若 `IterateOrder` 取值 `ORDER_M`，则 N 方向循环搬运（在 `singleCoreN` 大于 `baseN` 场景可能有性能提升），即 B 矩阵的 MTE1 搬运并行；
  - 若 `IterateOrder` 取值 `ORDER_N`，则 M 方向循环搬运（在 `singleCoreM` 大于 `baseM` 场景可能有性能提升），即 A 矩阵的 MTE1 搬运并行；
  - 不能同时使能 M 方向和 N 方向循环搬运；
- **注**：
  - Norm 模板的 Batch Matmul 场景或者 MDL 模板中，`singleCoreK > baseK` 时，不能使能 `ScheduleType::OUTER_PRODUCT` 取值，需使用默认模式。
  - Norm 模板或 MDL 模板的 Matmul 场景，仅支持在纯 Cube 模式（只有矩阵计算）下配置 `ScheduleType::OUTER_PRODUCT`。
  - MDL 模板仅在调用 `IterateAll` 计算的场景支持配置 `ScheduleType::OUTER_PRODUCT`。
  - 仅在 C 矩阵输出至 GM 时，支持配置 `ScheduleType::OUTER_PRODUCT`。
- **注意**：
  - Atlas A3 训练系列产品/Atlas A3 推理系列产品支持该参数。
  - Atlas A2 训练系列产品/Atlas A2 推理系列产品支持该参数。
  - Atlas 推理系列产品 AI Core 不支持该参数。
  - Atlas 200I/500 A2 推理产品不支持该参数。
- **支持模板**：Norm、MDL

### enableDoubleCache

- **说明**：开启 IBShare 模板后，在 L1 Buffer 上是否同时缓存两块数据。参数取值如下：
  - `false`：L1 Buffer 上同时缓存一块数据（默认值）。
  - `true`：使能 L1 Buffer 上同时缓存两块数据。
- **注意**：该参数取值为 `true` 时，需要控制基本块大小，防止两块数据的缓存超过 L1 Buffer 大小限制。
- **支持模板**：IBShare

### isBiasBatch

- **说明**：批量多 Batch 的 Matmul 场景，即 BatchMatmul 场景，Bias 的大小是否带有 Batch 轴。参数取值如下：
  - `true`：Bias 带有 Batch 轴，Bias 大小为 `Batch * N`（默认值）。
  - `false`：Bias 不带 Batch 轴，Bias 大小为 N，多 Batch 计算 Matmul 时，会复用 Bias。
- **注意**：
  - `BatchMode::SINGLE_LARGE_THAN_L1` 场景仅支持设置为 `true`。
  - Atlas A2 训练系列产品/Atlas A2 推理系列产品支持该参数。
  - Atlas A3 训练系列产品/Atlas A3 推理系列产品支持该参数。
  - Atlas 推理系列产品 AI Core 不支持设置为 `false`。
  - Atlas 200I/500 A2 推理产品不支持设置为 `false`。
- **支持模板**：Norm

### enableStaticPadZeros

- **说明**：使用常量化的 Tiling 参数时，在左矩阵和右矩阵搬运到 L1 Buffer 的过程中，是否自动按照常量化的 `singleM`/`singleN`/`singleK` 及 `baseM`/`baseN`/`baseK` 大小补零。关于常量化 Tiling 参数的详细内容请参考 `GetMatmulApiTiling`。
  仅支持 GM 输入的 ND2NZ 格式的补零，其他场景需要用户自行补零。参数取值如下：
  - `false`：搬运时不自动补零，需要用户自行补零（默认值）。
  - `true`：搬运时按照常量化的 `singleM`/`singleN`/`singleK` 及 `baseM`/`baseN`/`baseK` 大小自动补零。
- **支持模板**：Norm、MDL

### isPartialOutput

- **说明**：是否开启 PartialOutput 功能，即控制 Matmul 顺序输出 K 方向的基本块计算方式：Matmul 一次 Iterate 计算的 K 轴是否进行累加计算。参数取值如下：
  - `true`：开启 PartialOutput 功能，一次 Iterate 的 K 轴不进行累加计算，Matmul 每次计算输出局部 `baseK` 的 `baseM * baseN` 大小的矩阵分片。
  - `false`：不开启 PartialOutput 功能，一次 Iterate 的 K 轴进行累加计算，Matmul 每次计算输出 `SingleCoreK` 长度的 `baseM * baseN` 大小的矩阵分片。
- **注意**：
  - Atlas A3 训练系列产品/Atlas A3 推理系列产品支持该参数。
  - Atlas A2 训练系列产品/Atlas A2 推理系列产品支持该参数。
  - Atlas 推理系列产品 AI Core 不支持该参数。
  - Atlas 200I/500 A2 推理产品不支持该参数。
- **支持模板**：MDL

### enableMixDualMaster

- **说明**：是否使能 MixDualMaster（双主模式）。区别于 MIX 模式（包含矩阵计算和矢量计算）通过消息机制驱动 AIC 运行，双主模式为 AIC 和 AIV 独立运行代码，不依赖消息驱动，用于提升性能。该参数默认值为 `false`，仅能在以下场景设置为 `true`：
  - 核函数的类型为 MIX，同时 AIC 核数 : AIV 核数为 1:1。
  - 核函数的类型为 MIX，同时 AIC 核数 : AIV 核数为 1:2，且 A 矩阵和 B 矩阵同时使能 IBSHARE 参数。
- **注意**，使能 MixDualMaster 场景，需要满足：
  - 同一算子中所有 Matmul 对象的该参数取值必须保持一致。
  - A/B/Bias 矩阵只支持从 GM 搬入。
  - 获取矩阵计算结果只支持调用 `IterateAll` 接口输出到 `GlobalTensor`，即计算结果放置于 Global Memory 的地址，不能调用 `GetTensorC` 等接口获取结果。
  该参数的具体使用请参考使能双主模式的算子样例。
- **注意**：
  - Atlas A3 训练系列产品/Atlas A3 推理系列产品支持该参数。
  - Atlas A2 训练系列产品/Atlas A2 推理系列产品支持该参数。
  - Atlas 推理系列产品 AI Core 不支持该参数。
  - Atlas 200I/500 A2 推理产品不支持该参数。
- **支持模板**：Norm

### isA2B2Shared

- **说明**：是否开启 A2 和 B2 的全局管理，即控制所有 Matmul 对象是否共用 A2 和 B2 的 double buffer 机制。该配置为全局配置，所有 Matmul 对象取值必须保持一致。注意，开启时，A 矩阵、B 矩阵的基本块大小均不能超过 32KB。参数取值如下：
  - `true`：开启。
  - `false`：关闭（默认值）。
- **注意**：
  - Atlas A3 训练系列产品/Atlas A3 推理系列产品支持该参数。
  - Atlas A2 训练系列产品/Atlas A2 推理系列产品支持该参数。
  - Atlas 推理系列产品 AI Core 不支持该参数。
  - Atlas 200I/500 A2 推理产品不支持该参数。
  该参数取值为 `true` 时，建议同时设置 `enUnitFlag` 参数为 `true`，使搬运与计算流水并行，提高性能。该参数的使用样例请参考 Matmul A2 和 B2 全局管理样例。
- **支持模板**：Norm、MDL

### isEnableChannelSplit

- **说明**：是否使能 `channel_split` 功能。正常情况下，Matmul 计算出的 `CubeFormat::NZ` 格式的 C 矩阵分形为 16*16，假设此时的分形个数为 x，`channel_split` 功能是使获得的 C 矩阵分形为 16*8，同时分形个数变为 2x。注意，当前仅在 Matmul 计算结果 C 矩阵的 Format 为 `CubeFormat::NZ`，TYPE 为 `float` 类型，输出到 Global Memory 的场景，支持使能该参数。参数取值如下：
  - `false`：默认值，不使能 `channel_split` 功能，输出的分形为 16*16。
  - `true`：使能 `channel_split` 功能，输出的分形为 16*8。
- **注意**：
  - Atlas A3 训练系列产品/Atlas A3 推理系列产品支持该参数。
  - Atlas A2 训练系列产品/Atlas A2 推理系列产品支持该参数。
  - Atlas 推理系列产品 AI Core 不支持该参数。
  - Atlas 200I/500 A2 推理产品不支持该参数。
- **支持模板**：所有模板

### enableKdimReorderLoad

- **说明**：是否使能 K 轴错峰加载数据。基于相同 Tiling 参数，执行 Matmul 计算时，如果多核的左矩阵或者右矩阵相同，且存储于 Global Memory，多个核一般会同时访问相同地址以加载矩阵数据，引发同地址访问冲突，影响性能。使能该参数后，多核执行 Matmul 时，将尽量在相同时间访问矩阵的不同 Global Memory 地址，减少地址访问冲突概率，提升性能。该参数功能只支持 MDL 模板，建议 K 轴较大且左矩阵和右矩阵均非全载场景使能参数。参数取值如下，具体样例请参考 K 轴错峰加载数据的算子样例。
  - `false`：默认值，关闭 K 轴错峰加载数据的功能。
  - `true`：开启 K 轴错峰加载数据的功能。
- **注意**：
  - Atlas A3 训练系列产品/Atlas A3 推理系列产品支持该参数。
  - Atlas A2 训练系列产品/Atlas A2 推理系列产品支持该参数。
  - Atlas 推理系列产品 AI Core 不支持该参数。
  - Atlas 200I/500 A2 推理产品不支持该参数。
- **支持模板**：MDL

### isCO1Shared

- **说明**：是否使能 CO1 内存共享，由该参数与 `sharedCO1BufferSize` 参数指定 CO1 划分块数，缓存到 CO1 中的数据块数不能超过 CO1 划分的块数，即未被 `GetTensorC` 获取的 Iterate 计算生成的结果个数不能超过 CO1 划分的块数。该配置为全局配置，所有 Matmul 对象取值必须保持一致。参数取值如下：
  - `true`：开启 CO1 内存共享。
  - `false`：默认值，关闭 CO1 内存共享。
- **注意**：
  - Atlas A3 训练系列产品/Atlas A3 推理系列产品不支持该参数。
  - Atlas A2 训练系列产品/Atlas A2 推理系列产品不支持该参数。
  - Atlas 推理系列产品 AI Core 不支持该参数。
  - Atlas 200I/500 A2 推理产品不支持该参数。
- **支持模板**：Norm、IBShare

### sharedCO1BufferSize

- **说明**：指定 CO1 共享的一份 Buffer 大小。`uint32_t` 类型，支持的取值为 `32*1024`、`64*1024`、`128*1024`。
- **注意**：
  - Atlas A3 训练系列产品/Atlas A3 推理系列产品不支持该参数。
  - Atlas A2 训练系列产品/Atlas A2 推理系列产品不支持该参数。
  - Atlas 推理系列产品 AI Core 不支持该参数。
  - Atlas 200I/500 A2 推理产品不支持该参数。
- **支持模板**：Norm、IBShare
