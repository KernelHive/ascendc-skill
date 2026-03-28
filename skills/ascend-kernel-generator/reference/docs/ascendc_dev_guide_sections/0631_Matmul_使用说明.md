###### Matmul 使用说明

Ascend C 提供一组 Matmul 高阶 API，方便用户快速实现 Matmul 矩阵乘法的运算操作。

Matmul 的计算公式为：`C = A * B + Bias`，其示意图如下。

- **A、B** 为源操作数，A 为左矩阵，形状为 `[M, K]`；B 为右矩阵，形状为 `[K, N]`。
- **C** 为目的操作数，存放矩阵乘结果的矩阵，形状为 `[M, N]`。
- **Bias** 为矩阵乘偏置，形状为 `[1, N]`。对 `A*B` 结果矩阵的每一行都采用该 Bias 进行偏置。

> 图 15-37 Matmul 矩阵乘示意图

## 说明

下文中提及的：
- **M 轴方向**：即为 A 矩阵纵向；
- **K 轴方向**：即为 A 矩阵横向或 B 矩阵纵向；
- **N 轴方向**：即为 B 矩阵横向；
- **尾轴**：即为矩阵最后一个维度。

Kernel 侧实现 Matmul 矩阵乘运算的步骤概括为：

1. 创建 Matmul 对象
2. 初始化操作
3. 设置左矩阵 A、右矩阵 B、Bias
4. 完成矩阵乘操作
5. 结束矩阵乘操作

## 使用步骤

### 步骤 1：创建 Matmul 对象

创建 Matmul 对象的示例如下：

- **默认为 MIX 模式**（包含矩阵计算和矢量计算），该场景下，不能定义 `ASCENDC_CUBE_ONLY` 宏。
- **纯 Cube 模式**（只有矩阵计算）场景下，需要在代码中定义 `ASCENDC_CUBE_ONLY` 宏。

```cpp
// 纯cube模式（只有矩阵计算）场景下，需要设置该代码宏，并且必须在#include "lib/matmul_intf.h"之前设置
// #define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"

typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> aType;
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> bType;
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> cType;
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> biasType;
AscendC::Matmul<aType, bType, cType, biasType> mm;
```

创建对象时需要传入 A、B、C、Bias 的参数类型信息，类型信息通过 `MatmulType` 来定义，包括：内存逻辑位置、数据格式、数据类型。

```cpp
template <AscendC::TPosition POSITION, CubeFormat FORMAT, typename TYPE, bool ISTRANS = false,
LayoutMode LAYOUT = LayoutMode::NONE, bool IBSHARE = false> struct MatmulType {
constexpr static AscendC::TPosition pos = POSITION;
constexpr static CubeFormat format = FORMAT;
using T = TYPE;
constexpr static bool isTrans = ISTRANS;
constexpr static LayoutMode layout = LAYOUT;
constexpr static bool ibShare = IBSHARE;
};
```

#### MatmulType 参数说明

| 参数 | 说明 |
|------|------|
| **POSITION** | 内存逻辑位置。<br>• **Atlas A3 训练系列产品/Atlas A3 推理系列产品**：<br>  - A 矩阵：`TPosition::GM`、`TPosition::VECOUT`、`TPosition::TSCM`<br>  - B 矩阵：`TPosition::GM`、`TPosition::VECOUT`、`TPosition::TSCM`<br>  - Bias：`TPosition::GM`、`TPosition::VECOUT`<br>  - C 矩阵：`TPosition::GM`、`TPosition::VECIN`、`TPosition::CO1`<br>• **Atlas A2 训练系列产品/Atlas A2 推理系列产品**：<br>  - A 矩阵：`TPosition::GM`、`TPosition::VECOUT`、`TPosition::TSCM`<br>  - B 矩阵：`TPosition::GM`、`TPosition::VECOUT`、`TPosition::TSCM`<br>  - Bias：`TPosition::GM`、`TPosition::VECOUT`<br>  - C 矩阵：`TPosition::GM`、`TPosition::VECIN`、`TPosition::CO1`<br>  注意：C 矩阵设置为 `TPosition::CO1` 时，C 矩阵的数据排布格式仅支持 `CubeFormat::NZ`，C 矩阵的数据类型仅支持 `float`、`int32_t`。<br>• **Atlas 推理系列产品 AI Core**：<br>  - A 矩阵：`TPosition::GM`、`TPosition::VECOUT`<br>  - B 矩阵：`TPosition::GM`、`TPosition::VECOUT`<br>  - Bias：`TPosition::GM`、`TPosition::VECOUT`<br>  - C 矩阵：`TPosition::GM`、`TPosition::VECIN`<br>• **Atlas 200I/500 A2 推理产品**：<br>  - A 矩阵：`TPosition::GM`<br>  - B 矩阵：`TPosition::GM`<br>  - Bias：`TPosition::GM`<br>  - C 矩阵：`TPosition::GM` |
| **FORMAT** | 数据的物理排布格式，详细介绍请参考数据格式。<br>• **Atlas A3 训练系列产品/Atlas A3 推理系列产品**：<br>  - A 矩阵：`CubeFormat::ND`、`CubeFormat::NZ`、`CubeFormat::VECTOR`<br>  - B 矩阵：`CubeFormat::ND`、`CubeFormat::NZ`<br>  - Bias：`CubeFormat::ND`<br>  - C 矩阵：`CubeFormat::ND`、`CubeFormat::NZ`、`CubeFormat::ND_ALIGN`<br>• **Atlas A2 训练系列产品/Atlas A2 推理系列产品**：<br>  - A 矩阵：`CubeFormat::ND`、`CubeFormat::NZ`、`CubeFormat::VECTOR`<br>  - B 矩阵：`CubeFormat::ND`、`CubeFormat::NZ`<br>  - Bias：`CubeFormat::ND`<br>  - C 矩阵：`CubeFormat::ND`、`CubeFormat::NZ`、`CubeFormat::ND_ALIGN`<br>• **Atlas 推理系列产品 AI Core**：<br>  - A 矩阵：`CubeFormat::ND`、`CubeFormat::NZ`<br>  - B 矩阵：`CubeFormat::ND`、`CubeFormat::NZ`<br>  - Bias：`CubeFormat::ND`<br>  - C 矩阵：`CubeFormat::ND`、`CubeFormat::NZ`、`CubeFormat::ND_ALIGN`<br>  注意：C 矩阵设置为 `CubeFormat::ND` 时，要求尾轴 32 字节对齐，比如数据类型是 half 的情况下，N 要求是 16 的倍数。<br>• **Atlas 200I/500 A2 推理产品**：<br>  - A 矩阵：`CubeFormat::ND`、`CubeFormat::NZ`<br>  - B 矩阵：`CubeFormat::ND`、`CubeFormat::NZ`<br>  - Bias：`CubeFormat::ND`<br>  - C 矩阵：`CubeFormat::ND`、`CubeFormat::NZ`<br>  注意：C 矩阵设置为 `TPosition::VECIN` 或 `TPosition::TSCM`、`CubeFormat::ND` 时，要求尾轴 32 字节对齐，比如数据类型是 half 的情况下，N 要求是 16 的倍数；C 矩阵设置为 `TPosition::VECIN` 或 `TPosition::TSCM`、`CubeFormat::NZ` 时，N 要求是 16 的倍数。<br>关于 `CubeFormat::NZ` 格式的 A 矩阵、B 矩阵、C 矩阵的对齐约束，请参考表 15-590。 |
| **TYPE** | 数据类型。<br>• **Atlas A3 训练系列产品/Atlas A3 推理系列产品**：<br>  - A 矩阵：`half`、`float`、`bfloat16_t`、`int8_t`、`int4b_t`<br>  - B 矩阵：`half`、`float`、`bfloat16_t`、`int8_t`、`int4b_t`<br>  - Bias：`half`、`float`、`int32_t`<br>  - C 矩阵：`half`、`float`、`bfloat16_t`、`int32_t`、`int8_t`<br>• **Atlas A2 训练系列产品/Atlas A2 推理系列产品**：<br>  - A 矩阵：`half`、`float`、`bfloat16_t`、`int8_t`、`int4b_t`<br>  - B 矩阵：`half`、`float`、`bfloat16_t`、`int8_t`、`int4b_t`<br>  - Bias：`half`、`float`、`int32_t`<br>  - C 矩阵：`half`、`float`、`bfloat16_t`、`int32_t`、`int8_t`<br>• **Atlas 推理系列产品 AI Core**：<br>  - A 矩阵：`half`、`int8_t`<br>  - B 矩阵：`half`、`int8_t`<br>  - Bias：`float`、`int32_t`<br>  - C 矩阵：`half`、`float`、`int8_t`、`int32_t`<br>• **Atlas 200I/500 A2 推理产品**：<br>  - A 矩阵：`half`、`float`、`bfloat16_t`、`int8_t`<br>  - B 矩阵：`half`、`float`、`bfloat16_t`、`int8_t`<br>  - Bias：`half`、`float`、`int32_t`<br>  - C 矩阵：`half`、`float`、`bfloat16_t`、`int32_t`<br>注意：除 B 矩阵为 `int8_t` 数据类型外，A 矩阵和 B 矩阵数据类型需要一致，具体数据类型组合关系请参考表 15-589。A 矩阵和 B 矩阵为 `int4b_t` 数据类型时，矩阵内轴的数据个数必须为偶数。例如，A 矩阵为 `int4b_t` 数据类型且不转置时，`singleCoreK` 必须是偶数。关于 `int4b_t` 数据类型的使用样例请参考 Int4 类型输入的 Matmul 算子样例。 |
| **ISTRANS** | 是否开启支持矩阵转置的功能。<br>• `true`：开启支持矩阵转置的功能，运行时可以分别通过 `SetTensorA` 和 `SetTensorB` 中的 `isTransposeA`、`isTransposeB` 参数设置 A、B 矩阵是否转置。若设置 A、B 矩阵转置，Matmul 会认为 A 矩阵形状为 `[K, M]`，B 矩阵形状为 `[N, K]`。<br>• `false`：默认值，不开启支持矩阵转置的功能，通过 `SetTensorA` 和 `SetTensorB` 不能设置 A、B 矩阵的转置情况。Matmul 会认为 A 矩阵形状为 `[M, K]`，B 矩阵形状为 `[K, N]`。<br>注意：由于 L1 Buffer 上的矩阵数据有分形对齐的约束，A、B 矩阵转置和不转置时所需的 L1 空间可能不相同，在开启支持矩阵转置功能时，必须保证按照 Matmul Tiling 参数申请的 L1 空间不超过 L1 Buffer 的规格，判断方式为 `(depthA1*Ceil(baseM/c0Size)*baseK + depthB1*Ceil(baseN/c0Size)*baseK) * db * sizeof(dtype) < L1Size`，`db` 表示 L1 是否开启 double buffer，取值 1（不开启 double buffer）或 2（开启 double buffer），其余参数的含义请参考表 15-642。 |
| **LAYOUT** | 表征数据的排布。<br>• `NONE`：默认值，表示不使用 BatchMatmul；其他选项表示使用 BatchMatmul。<br>• `NORMAL`：BMNK 的数据排布格式，具体可参考 `IterateBatch` 中对该数据排布的介绍。<br>• `BSNGD`：原始 BSH shape 做 reshape 后的数据排布，具体可参考 `IterateBatch` 中对该数据排布的介绍。<br>• `SBNGD`：原始 SBH shape 做 reshape 后的数据排布，具体可参考 `IterateBatch` 中对该数据排布的介绍。<br>• `BNGS1S2`：一般为前两种数据排布进行矩阵乘的输出，S1S2 数据连续存放，一个 S1S2 为一个 batch 的计算数据，具体可参考 `IterateBatch` 中对该数据排布的介绍。 |
| **IBSHARE** | 是否使能 IBShare（IntraBlock Share）。IBShare 的功能是能够复用 L1 Buffer 上相同的 A 矩阵或 B 矩阵数据，复用的矩阵必须在 L1 Buffer 上全载。A 矩阵和 B 矩阵仅有一个使能 IBShare 的场景，与 IBShare 模板配合使用，具体参数设置详见表 15-593。<br>注意：A 矩阵和 B 矩阵同时使能 IBShare 的场景，表示 L1 Buffer 上的 A 矩阵和 B 矩阵同时复用，需要满足：<br>• 同一算子中其它 Matmul 对象的 A 矩阵和 B 矩阵也必须同时使能 IBShare；<br>• Atlas A2 训练系列产品/Atlas A2 推理系列产品，获取矩阵计算结果时，只支持输出到 GlobalTensor，即计算结果放置于 Global Memory 的地址；<br>• Atlas A3 训练系列产品/Atlas A3 推理系列产品，获取矩阵计算结果时，只支持输出到 GlobalTensor，即计算结果放置于 Global Memory 的地址。<br>支持平台：<br>• Atlas A3 训练系列产品/Atlas A3 推理系列产品 ✅<br>• Atlas A2 训练系列产品/Atlas A2 推理系列产品 ✅<br>• Atlas 推理系列产品 AI Core ❌<br>• Atlas 200I/500 A2 推理产品 ❌<br>该参数使用样例请参考 MatmulABshare 样例、A、B 矩阵均使能 IBShare 样例、仅 B 矩阵使能 IBShare 样例。 |

#### Matmul 输入输出数据类型的支持列表

| A 矩阵 | B 矩阵 | Bias | C 矩阵 | 支持平台 |
|--------|--------|------|--------|----------|
| `float` | `float` | `float` / `half` | `float` | • Atlas A3 训练系列产品/Atlas A3 推理系列产品<br>• Atlas A2 训练系列产品/Atlas A2 推理系列产品<br>• Atlas 200I/500 A2 推理产品 |
| `half` | `half` | `float` | `float` | • Atlas A3 训练系列产品/Atlas A3 推理系列产品<br>• Atlas A2 训练系列产品/Atlas A2 推理系列产品<br>• Atlas 推理系列产品 AI Core<br>• Atlas 200I/500 A2 推理产品 |
| `half` | `half` | `half` | `float` | • Atlas A3 训练系列产品/Atlas A3 推理系列产品<br>• Atlas A2 训练系列产品/Atlas A2 推理系列产品<br>• Atlas 200I/500 A2 推理产品 |
| `int8_t` | `int8_t` | `int32_t` | `int32_t` / `half` | • Atlas A3 训练系列产品/Atlas A3 推理系列产品<br>• Atlas A2 训练系列产品/Atlas A2 推理系列产品<br>• Atlas 推理系列产品 AI Core<br>• Atlas 200I/500 A2 推理产品 |
| `int4b_t` | `int4b_t` | `int32_t` | `int32_t` / `half` | • Atlas A3 训练系列产品/Atlas A3 推理系列产品<br>• Atlas A2 训练系列产品/Atlas A2 推理系列产品 |
| `bfloat16_t` | `bfloat16_t` | `float` | `float` | • Atlas A3 训练系列产品/Atlas A3 推理系列产品<br>• Atlas A2 训练系列产品/Atlas A2 推理系列产品<br>• Atlas 200I/500 A2 推理产品 |
| `bfloat16_t` | `bfloat16_t` | `half` | `float` | • Atlas A3 训练系列产品/Atlas A3 推理系列产品<br>• Atlas A2 训练系列产品/Atlas A2 推理系列产品 |
| `half` | `half` | `float` | `int8_t` | • Atlas A3 训练系列产品/Atlas A3 推理系列产品<br>• Atlas A2 训练系列产品/Atlas A2 推理系列产品 |
| `bfloat16_t` | `bfloat16_t` | `float` | `int8_t` | • Atlas A3 训练系列产品/Atlas A3 推理系列产品<br>• Atlas A2 训练系列产品/Atlas A2 推理系列产品 |
| `int8_t` | `int8_t` | `int32_t` | `int8_t` | • Atlas A3 训练系列产品/Atlas A3 推理系列产品<br>• Atlas A2 训练系列产品/Atlas A2 推理系列产品<br>• Atlas 推理系列产品 AI Core |
| `half` | `half` | `float` | `half` | • Atlas A3 训练系列产品/Atlas A3 推理系列产品<br>• Atlas A2 训练系列产品/Atlas A2 推理系列产品<br>• Atlas 推理系列产品 AI Core<br>• Atlas 200I/500 A2 推理产品 |
| `half` | `half` | `half` | `half` | • Atlas A3 训练系列产品/Atlas A3 推理系列产品<br>• Atlas A2 训练系列产品/Atlas A2 推理系列产品<br>• Atlas 200I/500 A2 推理产品 |
| `bfloat16_t` | `bfloat16_t` | `float` | `bfloat16_t` | • Atlas A3 训练系列产品/Atlas A3 推理系列产品<br>• Atlas A2 训练系列产品/Atlas A2 推理系列产品<br>• Atlas 200I/500 A2 推理产品 |
| `half` | `int8_t` | `float` | `float` | • Atlas 推理系列产品 AI Core |

### 步骤 2：初始化操作

```cpp
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling); // 初始化matmul对象，参数含义请参考 REGIST_MATMUL_OBJ 章节
```

### 步骤 3：设置左矩阵 A、右矩阵 B、Bias

```cpp
mm.SetTensorA(gm_a); // 设置左矩阵A
mm.SetTensorB(gm_b); // 设置右矩阵B
mm.SetBias(gm_bias); // 设置Bias

// Atlas 推理系列产品AI Core上需要额外调用SetLocalWorkspace接口设置计算所需的UB空间
mm.SetLocalWorkspace(usedUbBufLen);
```

### 步骤 4：完成矩阵乘操作

用户可以选择以下三种调用方式之一。

#### 方式一：调用 Iterate 完成单次迭代计算

```cpp
// API接口内部会进行循环结束条件判断处理
while (mm.Iterate()) {
    mm.GetTensorC(gm_c);
}
```

#### 方式二：调用 IterateAll 完成单核上所有数据的计算

```cpp
mm.IterateAll(gm_c);
```

#### 方式三：使用 CO1 内存并调用 Fixpipe 搬运结果

在此种调用方式下，创建 Matmul 对象时，必须定义 C 矩阵的内存逻辑位置为 `TPosition::CO1`、数据排布格式为 `CubeFormat::NZ`、数据类型为 `float` 或 `int32_t`。

> **注意**：
> - Atlas 推理系列产品 AI Core 暂不支持该方式。
> - Atlas 200I/500 A2 推理产品暂不支持该方式。

```cpp
// 定义C矩阵的类型信息
typedef AscendC::MatmulType<AscendC::TPosition::CO1, CubeFormat::NZ, float> cType;
// 创建Matmul对象
AscendC::Matmul<aType, bType, cType, biasType> mm;

// 用户提前申请CO1的内存l0cTensor
TQue<TPosition::CO1, 1> CO1_;
// 128 * 1024为申请的CO1内存大小
GetTPipePtr()->InitBuffer(CO1_, 1, 128 * 1024);
// L0cT为C矩阵的数据类型。
// A矩阵数据类型是int8_t或int4b_t时，C矩阵的数据类型是int32_t。
// A矩阵数据类型是half、float或bfloat16_t时，C矩阵的数据类型是float。
LocalTensor<L0cT> l0cTensor = CO1_.template AllocTensor<L0cT>();

// 将l0cTensor作为入参传入Iterate，矩阵乘结果输出到用户申请的l0cTensor上
mm.Iterate(false, l0cTensor);

// 调用Fixpipe接口将CO1上的计算结果搬运到GM
FixpipeParamsV220 params;
params.nSize = nSize;
params.mSize = mSize;
params.srcStride = srcStride;
params.dstStride = dstStride;
CO1_.EnQue(l0cTensor);
CO1_.template DeQue<L0cT>();
Fixpipe<cType, L0cT, CFG_ROW_MAJOR>(gm[dstOffset], l0cTensor, params);

//释放CO1内存
CO1_.FreeTensor(l0cTensor);
```

### 步骤 5：结束矩阵乘操作

```cpp
mm.End();
```

## CubeFormat::NZ 格式的矩阵对齐要求

| 源/目的操作数 | 外轴 | 内轴 |
|---------------|------|------|
| A 矩阵 / B 矩阵 | 16 的倍数 | C0_size 的倍数 |
| C 矩阵 | 16 的倍数 | 16 的倍数 |
| C 矩阵（使能 channel_split 功能） | 16 的倍数 | C0_size 的倍数 |
| C 矩阵（不使能 channel_split 功能） | 16 的倍数 | `float` / `int32_t`：16 的倍数<br>`half` / `bfloat16_t` / `int8_t`：C0_size 的倍数 |

> **注 1**：`float` / `int32_t` 数据类型的 `C0_size` 为 8，`half` / `bfloat16_t` 数据类型的 `C0_size` 为 16，`int8_t` 数据类型的 `C0_size` 为 32，`int4b_t` 数据类型的 `C0_size` 为 64。
>
> **注 2**：channel_split 功能通过 `MatmulConfig` 中的 `isEnableChannelSplit` 参数配置，具体内容请参考 `MatmulConfig`。

## 需要包含的头文件

```cpp
#include "lib/matmul/matmul_intf.h"
```

## 实现原理

以输入矩阵 A（GM, ND, half）、矩阵 B（GM, ND, half），输出矩阵 C（GM, ND, float），无 Bias 场景为例，其中（GM, ND, half）表示数据存放在 GM 上，数据格式为 ND，数据类型为 half，描述 Matmul 高阶 API 典型场景的内部算法框图，如下图所示。

> 图 15-38 Matmul 算法框图

计算过程分为如下几步：

1. **数据从 GM 搬到 A1**：`DataCopy` 每次从矩阵 A，搬出一个 `stepM*baseM*stepKa*baseK` 的矩阵块 a1，循环多次完成矩阵 A 的搬运；
   **数据从 GM 搬到 B1**：`DataCopy` 每次从矩阵 B，搬出一个 `stepKb*baseK*stepN*baseN` 的矩阵块 b1，循环多次完成矩阵 B 的搬运；
2. **数据从 A1 搬到 A2**：`LoadData` 每次从矩阵块 a1，搬出一个 `baseM * baseK` 的矩阵块 a0；
   **数据从 B1 搬到 B2，并完成转置**：`LoadData` 每次从矩阵块 b1，搬出一个 `baseK * baseN` 的矩阵块，并将其转置为 `baseN * baseK` 的矩阵块 b0；
3. **矩阵乘**：每次完成一个矩阵块 `a0 * b0` 的计算，得到 `baseM * baseN` 的矩阵块 co1；
4. **数据从矩阵块 co1 搬到矩阵块 co2**：`DataCopy` 每次搬运一块 `baseM * baseN` 的矩阵块 co1 到 `singleCoreM * singleCoreN` 的矩阵块 co2 中；
5. 重复 2-4 步骤，完成矩阵块 `a1 * b1` 的计算；
6. **数据从矩阵块 co2 搬到矩阵块 C**：`DataCopy` 每次搬运一块 `singleCoreM * singleCoreN` 的矩阵块 co2 到矩阵块 C 中；
7. 重复 1-6 步骤，完成矩阵 `A * B = C` 的计算。

> **注意**：`stepM`、`baseM` 等参数的含义请参考 Tiling 参数。
