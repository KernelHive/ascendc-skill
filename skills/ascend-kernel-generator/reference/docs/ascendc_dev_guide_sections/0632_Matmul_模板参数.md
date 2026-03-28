###### Matmul 模板参数

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

## 功能说明

创建Matmul对象时需要传入：

- **A、B、C、Bias的参数类型信息**：类型信息通过MatmulType来定义，包括：内存逻辑位置、数据格式、数据类型、是否转置、数据排布和是否使能L1复用。
- **MatmulConfig信息（可选）**：用于配置Matmul模板信息以及相关的配置参数。不配置默认使能Norm模板。
  - 针对Atlas 200I/500 A2 推理产品，当前只支持使能默认的Norm模板。
- **MatmulCallBackFunc回调函数信息（可选）**：用于配置左右矩阵从GM拷贝到A1/B1、计算结果从CO1拷贝到GM的自定义函数。当前支持如下产品型号：
  - Atlas A3 训练系列产品/Atlas A3 推理系列产品
  - Atlas A2 训练系列产品/Atlas A2 推理系列产品
- **MatmulPolicy信息（可选）**：用于配置Matmul可拓展模块策略。不配置使能默认模板策略。当前支持如下产品型号：
  - Atlas A3 训练系列产品/Atlas A3 推理系列产品
  - Atlas A2 训练系列产品/Atlas A2 推理系列产品
  - Atlas 200I/500 A2 推理产品
  - Atlas 推理系列产品AI Core

## 函数原型

```cpp
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE = C_TYPE, const auto& MM_CFG = CFG_NORM, class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>, MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)>
using Matmul = AscendC::MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;
```

- **A_TYPE、B_TYPE、C_TYPE**：类型信息通过MatmulType来定义。
- **auto类型的参数MM_CFG（可选）**：
  - 支持MatmulConfig类型：Matmul模板信息，具体内容见MatmulConfig。
  - 支持MatmulApiStaticTiling类型：MatmulApiStaticTiling参数说明见表15-591。MatmulApiStaticTiling结构体中包括一组常量化Tiling参数和MatmulConfig结构。这种类型参数的定义方式为，通过调用MatmulConfig章节中介绍的获取模板的接口，指定(singleM, singleN, singleK, baseM, baseN, baseK)参数，获取自定义模板；将该模板传入GetMatmulApiTiling接口，得到常量化的参数。这种常量化的方式将得到MatmulApiStaticTiling结构体中定义的一组常量化参数，可以优化Matmul计算中的Scalar计算。当前支持定义为MatmulApiStaticTiling常量化的Tiling参数的模板有：Norm、IBShare、MDL模板。
- **MM_CB（可选）**：用于支持不同的搬入搬出需求，实现定制化的搬入搬出功能。具体内容见MatmulCallBackFunc。
- **MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)（可选）**：用于配置Matmul可拓展模块的策略。当前支持不配置该参数（使能默认模板策略）或者配置1个MatmulPolicy参数。

MATMUL_POLICY_DEFAULT_OF定义如下，用于简化MATMUL_POLICY的类型声明。该模板参数的详细使用方式请参考MatmulPolicy。

```cpp
#define MATMUL_POLICY_DEFAULT_OF(DEFAULT) \
template <const auto& = MM_CFG, typename ...> \
class MATMUL_POLICY = AscendC::Impl::Detail::DEFAULT
```

## 参数说明

### 表 15-591 MatmulApiStaticTiling 常量化 Tiling 参数说明

| 参数 | 数据类型 | 说明 |
|------|----------|------|
| M, N, Ka, Kb | int32_t | 与TCubeTiling结构体中各同名参数含义一致。本结构体中的参数是常量化后的常数值。 |
| singleCoreM | int32_t |  |
| singleCoreN | int32_t |  |
| singleCoreK | int32_t |  |
| baseM | int32_t |  |
| baseN | int32_t |  |
| baseK | int32_t |  |
| depthA1 | int32_t |  |
| depthB1 | int32_t |  |
| stepM | int32_t |  |
| stepN | int32_t |  |
| stepKa | int32_t |  |
| stepKb | int32_t |  |
| isBias | int32_t |  |
| transLength | int32_t |  |
| iterateOrder | int32_t |  |
| dbL0A | int32_t |  |
| dbL0B | int32_t |  |
| dbL0C | int32_t |  |
| shareMode | int32_t |  |
| shareL1Size | int32_t |  |
| shareL0CSize | int32_t |  |
| shareUbSize | int32_t |  |
| batchM | int32_t |  |
| batchN | int32_t |  |
| singleBatchM | int32_t |  |
| singleBatchN | int32_t |  |
| cfg | MatmulConfig | Matmul模板的参数配置 |

## 返回值说明

无

## 约束说明

无

## 调用示例

```cpp
//用户自定义回调函数
void DataCopyOut(const __gm__ void *gm, const LocalTensor<int8_t> &co1Local, const void *dataCopyOutParams, const uint64_t tilingPtr, const uint64_t dataPtr);
void CopyA1(const AscendC::LocalTensor<int8_t> &aMatrix, const __gm__ void *gm, int row, int col, int useM, int useK, const uint64_t tilingPtr, const uint64_t dataPtr);
void CopyB1(const AscendC::LocalTensor<int8_t> &bMatrix, const __gm__ void *gm, int row, int col, int useK, int useN, const uint64_t tilingPtr, const uint64_t dataPtr);

typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> aType;
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> bType;
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> cType;
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> biasType;

AscendC::Matmul<aType, bType, cType, biasType, CFG_MDL> mm1;

AscendC::MatmulConfig mmConfig{false, true, false, 128, 128, 64};
mmConfig.enUnitFlag = false;
AscendC::Matmul<aType, bType, cType, biasType, mmConfig> mm2;

AscendC::Matmul<aType, bType, cType, biasType, CFG_NORM, AscendC::MatmulCallBackFunc<DataCopyOut, CopyA1, CopyB1>> mm3;
```
