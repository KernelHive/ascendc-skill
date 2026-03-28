###### GetBasicConfig

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品AI Core | × |
| Atlas 推理系列产品Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

用于配置BasicBlock模板的参数，获取自定义BasicBlock模板。BasicBlock模板的介绍请参考表"模板特性"。

使用该接口时可以优先考虑使用模板常量化。相比BasicBlock模板可以实现baseM/N/K常量化，模板常量化可以在此基础上实现singleCoreM/N/K、baseM/N/K的常量化，模板常量化的具体使用方式请参考Matmul Tiling常量化。

## 函数原型

```cpp
__aicore__ constexpr MatmulConfig GetBasicConfig(
    const uint32_t basicM,
    const uint32_t basicN, 
    const uint32_t basicK,
    const bool intrinsicsLimit = false,
    const bool batchLoop = false,
    const BatchMode bmmMode = BatchMode::BATCH_LESS_THAN_L1
)
```

## 参数说明

本接口的所有参数用于设置MatmulConfig结构体中的参数，其中互相对应的参数的功能作用相同。

### 表 15-601 接口参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| basicM | 输入 | 用于设置参数basicM。<br>与TCubeTiling结构体中的baseM参数含义相同，Matmul计算时base块M轴长度，以元素为单位。 |
| basicN | 输入 | 用于设置参数basicN。<br>与TCubeTiling结构体中的baseN参数含义相同，Matmul计算时base块N轴长度，以元素为单位。 |
| basicK | 输入 | 用于设置参数basicK。<br>与TCubeTiling结构体中的baseK参数含义相同，Matmul计算时base块K轴长度，以元素为单位。 |
| intrinsicsLimit | 输入 | 用于设置参数intrinsicsCheck。<br>当左矩阵或右矩阵在单核上内轴（即尾轴）大于等于65535（元素个数）时，是否使能循环执行数据从Global Memory到L1 Buffer的搬入。例如，左矩阵A[M, K]，单核上的内轴数据singleCoreK大于65535，配置该参数为true后，API内部通过循环执行数据的搬入。参数取值如下：<br>● false：当左矩阵或右矩阵在单核上内轴大于等于65535时，不使能循环执行数据的搬入（默认值）<br>● true：当左矩阵或右矩阵在单核上内轴大于等于65535时，使能循环执行数据的搬入 |
| batchLoop | 输入 | 用于设置参数isNBatch。<br>是否多Batch输入多Batch输出。仅对BatchMatmul有效，使能该参数后，仅支持Norm模板，且需调用IterateNBatch实现多Batch输入多Batch输出。参数取值如下：<br>● false：不使能多Batch（默认值）<br>● true：使能多Batch |
| bmmMode | 输入 | 用于设置参数batchMode。该参数用于BatchMatmul场景，关于BatchMatmul的介绍请参考6.3.3.13 Batch Matmul基础功能。<br>BatchMatmul场景中Layout类型为NORMAL时，设置BatchMatmul输入A/B矩阵的多batch数据总和与L1 Buffer的大小关系。参数取值如下：<br>● BatchMode::BATCH_LESS_THAN_L1：多batch数据总和<L1 Buffer Size<br>● BatchMode::BATCH_LARGE_THAN_L1：多batch数据总和>L1 Buffer Size<br>● BatchMode::SINGLE_LARGE_THAN_L1：单batch数据总和>L1 Buffer Size |

## 返回值说明

MatmulConfig结构体。

## 约束说明

无

## 调用示例

BasicBlock模板的完整使用样例请参考basic_block_matmul。

```cpp
constexpr MatmulConfig MM_CFG = GetBasicConfig(128, 256, 64); // baseM, baseN, baseK
AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG> mm;
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
mm.SetBias(gm_bias);
mm.IterateAll(gm_c);
```
