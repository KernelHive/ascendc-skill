###### GetNormalConfig

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

用于配置Norm模板的参数，获取自定义Norm模板。Norm模板的介绍请参考表模板特性。

## 函数原型

```cpp
__aicore__ constexpr MatmulConfig GetNormalConfig(
    const bool intrinsicsLimit = false,
    const bool batchLoop = false,
    const bool isVecND2NZ = false,
    const BatchMode bmmMode = BatchMode::BATCH_LESS_THAN_L1,
    const bool isMsgReuse = true,
    const IterateOrder iterateOrder = IterateOrder::UNDEF,
    const ScheduleType scheduleType = ScheduleType::INNER_PRODUCT,
    const bool enUnitFlag = true,
    const bool enableMixDualMaster = false,
    const BatchOutMode bmmOutMode = BatchOutMode::SINGLE_BATCH
)
```

## 参数说明

本接口的所有参数用于设置MatmulConfig结构体中的参数，其中互相对应的参数的功能作用相同。

### 表 15-597 接口参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| intrinsicsLimit | 输入 | 用于设置参数intrinsicsCheck。<br>当左矩阵或右矩阵在单核上内轴（即尾轴）大于等于65535（元素个数）时，是否使能循环执行数据从Global Memory到L1 Buffer的搬入。例如，左矩阵A[M, K]，单核上的内轴数据singleCoreK大于65535，配置该参数为true后，API内部通过循环执行数据的搬入。参数取值如下：<br>● false：当左矩阵或右矩阵在单核上内轴大于等于65535时，不使能循环执行数据的搬入（默认值）<br>● true：当左矩阵或右矩阵在单核上内轴大于等于65535时，使能循环执行数据的搬入 |
| batchLoop | 输入 | 用于设置参数isNBatch。<br>是否多Batch输入多Batch输出。仅对BatchMatmul有效，使能该参数后，仅支持Norm模板，且需调用IterateNBatch实现多Batch输入多Batch输出。参数取值如下：<br>● false：不使能多Batch（默认值）<br>● true：使能多Batch |
| isVecND2NZ | 输入 | 用于设置参数enVecND2NZ。<br>使能通过vector指令进行ND2NZ。使能时需要设置SetLocalWorkspace。参数取值如下：<br>● false：不使能通过vector指令进行ND2NZ（默认值）<br>● true：使能通过vector指令进行ND2NZ<br>针对Atlas 推理系列产品AI Core，在Unified Buffer空间足够的条件下（Unified Buffer空间大于2倍TCubeTiling的transLength参数），建议优先使能该参数，搬运性能更好 |
| bmmMode | 输入 | 用于设置参数batchMode。该参数用于BatchMatmul场景，关于BatchMatmul的介绍请参考6.3.3.13 Batch Matmul基础功能。<br>BatchMatmul场景中Layout类型为NORMAL时，设置BatchMatmul输入A/B矩阵的多batch数据总和与L1 Buffer的大小关系。参数取值如下：<br>● BatchMode::BATCH_LESS_THAN_L1：多batch数据总和<L1 Buffer Size<br>● BatchMode::BATCH_LARGE_THAN_L1：多batch数据总和>L1 Buffer Size<br>● BatchMode::SINGLE_LARGE_THAN_L1：单batch数据总和>L1 Buffer Size |
| isMsgReuse | 输入 | 用于设置参数enableReuse。<br>SetSelfDefineData函数设置的回调函数中的dataPtr是否直接传递计算数据。若未调用SetSelfDefineData设置dataPtr，该参数仅支持默认值true。参数取值如下：<br>● true：直接传递计算数据，仅限单个值<br>● false：传递GM上存储的数据地址信息 |
| iterateOrder | 输入 | 用于设置参数iterateOrder。<br>Matmul做矩阵运算的循环迭代顺序，与表15-642中的iterateOrder参数含义相同。当ScheduleType参数取值为ScheduleType::OUTER_PRODUCT时，本参数生效。参数取值如下：<br>```cpp<br>enum class IterateOrder {<br>    ORDER_M = 0, // 先往M轴方向偏移再往N轴方向偏移<br>    ORDER_N,     // 先往N轴方向偏移再往M轴方向偏移<br>    UNDEF,       // 当前无效<br>};<br>```<br>注：Norm模板的Matmul场景、MDL模板使用时，若IterateOrder取值ORDER_M，TCubeTiling结构中的stepN需要大于1，IterateOrder取值ORDER_N时，TCubeTiling结构中的stepM需要大于1。<br>该参数的使用样例请参考M/N方向流水并行Matmul算子样例。<br>**产品支持情况：**<br>● Atlas A3 训练系列产品/Atlas A3 推理系列产品支持该参数<br>● Atlas A2 训练系列产品/Atlas A2 推理系列产品支持该参数<br>● Atlas 推理系列产品AI Core不支持该参数<br>● Atlas 200I/500 A2 推理产品不支持该参数 |
| scheduleType | 输入 | 用于设置参数scheduleType。<br>配置Matmul数据搬运模式。参数取值如下：<br>● ScheduleType::INNER_PRODUCT：默认模式，在K方向上做MTE1的循环搬运<br>● ScheduleType::OUTER_PRODUCT：在M或N方向上做MTE1的循环搬运；使能后，需要与IterateOrder参数配合使用<br>该配置当前只在BatchMatmul场景（使能Norm模板）或Matmul场景（使能MDL模板或Norm模板）生效。<br>● 若IterateOrder取值ORDER_M，则N方向循环搬运（在singleCoreN大于baseN场景可能有性能提升），即B矩阵的MTE1搬运并行<br>● 若IterateOrder取值ORDER_N，则M方向循环搬运（在singleCoreM大于baseM场景可能有性能提升），即A矩阵的MTE1搬运并行<br>● 不能同时使能M方向和N方向循环搬运<br>注：<br>● Norm模板的Batch Matmul场景或者MDL模板中，singleCoreK>baseK时，不能使能ScheduleType::OUTER_PRODUCT取值，需使用默认模式<br>● Norm模板或MDL模板的Matmul场景，仅支持在纯Cube模式（只有矩阵计算）下配置ScheduleType::OUTER_PRODUCT<br>● MDL模板仅在调用IterateAll计算的场景支持配置ScheduleType::OUTER_PRODUCT<br>● 仅在C矩阵输出至GM时，支持配置ScheduleType::OUTER_PRODUCT<br>**产品支持情况：**<br>● Atlas A3 训练系列产品/Atlas A3 推理系列产品支持该参数<br>● Atlas A2 训练系列产品/Atlas A2 推理系列产品支持该参数<br>● Atlas 推理系列产品AI Core不支持该参数<br>● Atlas 200I/500 A2 推理产品不支持该参数 |
| enUnitFlag | 输入 | 用于设置参数enUnitFlag。<br>使能UnitFlag功能，使计算与搬运流水并行，提高性能。Norm, IBShare下默认使能，MDL下默认不使能。参数取值如下：<br>● false：不使能UnitFlag功能<br>● true：使能UnitFlag功能 |
| enableMixDualMaster | 输入 | 用于设置参数enableMixDualMaster。<br>是否使能MixDualMaster（双主模式）。区别于MIX模式（包含矩阵计算和矢量计算）通过消息机制驱动AIC运行，双主模式为AIC和AIV独立运行代码，不依赖消息驱动，用于提升性能。该参数默认值为false，仅能在以下场景设置为true：<br>● 核函数的类型为MIX，同时AIC核数 : AIV核数为1:1<br>● 核函数的类型为MIX，同时AIC核数 : AIV核数为1:2，且A矩阵和B矩阵同时使能IBSHARE参数<br>注意，使能MixDualMaster场景，需要满足：<br>● 同一算子中所有Matmul对象的该参数取值必须保持一致<br>● A/B/Bias矩阵只支持从GM搬入<br>● 获取矩阵计算结果只支持调用IterateAll接口输出到GlobalTensor，即计算结果放置于Global Memory的地址，不能调用GetTensorC等接口获取结果<br>**产品支持情况：**<br>● Atlas A3 训练系列产品/Atlas A3 推理系列产品支持该参数<br>● Atlas A2 训练系列产品/Atlas A2 推理系列产品支持该参数<br>● Atlas 推理系列产品AI Core不支持该参数<br>● Atlas 200I/500 A2 推理产品不支持该参数 |
| bmmOutMode | 输入 | 预留参数 |

## 返回值说明

MatmulConfig结构体。

## 约束说明

无

## 调用示例

```cpp
constexpr MatmulConfig MM_CFG = GetNormalConfig();
AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG> mm;
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
mm.SetBias(gm_bias);
mm.IterateAll(gm_c);
```
