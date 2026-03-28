###### Conv3DBackpropFilter Tiling 侧接口

## 使用说明

Ascend C提供一组Conv3DBackpropFilter Tiling API，方便用户获取Conv3DBackpropFilter Kernel计算时所需的Tiling参数。用户只需要传入Input/GradOutput/GradWeight的Position位置、Format格式和DType数据类型及相关参数等信息，调用API接口，即可获取Init中TConv3DBpFilterTiling结构体中的相关参数。

Conv3DBackpropFilter Tiling API提供一个GetTiling接口获取Tiling参数，获取Tiling参数的流程如下：

1. 创建一个单核Tiling对象
2. 设置Input、GradOutput、GradWeight的参数类型信息以及Shape信息，如果存在Padding、Stride参数，通过SetPadding、SetStride接口设置
3. 调用GetTiling接口，获取Tiling信息

使用Conv3DBackpropFilter Tiling接口获取Tiling参数的样例如下：

```cpp
#include "tiling/conv_backprop/conv3d_bp_filter_tiling.h"

optiling::Conv3DBackpropFilterTilingData tilingData;
auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
ConvBackpropApi::Conv3dBpFilterTiling conv3dBpDwTiling(*ascendcPlatform);

conv3dBpDwTiling.SetWeightType(ConvCommonApi::TPosition::CO1,
ConvCommonApi::ConvFormat::FRACTAL_Z_3D,
ConvCommonApi::ConvDtype::FLOAT32);
conv3dBpDwTiling.SetInputType(ConvCommonApi::TPosition::GM,
ConvCommonApi::ConvFormat::NDC1HWC0,
ConvCommonApi::ConvDtype::FLOAT16);
conv3dBpDwTiling.SetGradOutptutType(ConvCommonApi::TPosition::GM,
ConvCommonApi::ConvFormat::NDC1HWC0,
ConvCommonApi::ConvDtype::FLOAT16);
conv3dBpDwTiling.SetGradOutputShape(n, c, d, h, w);
conv3dBpDwTiling.SetInputShape(c, d, h, w);
conv3dBpDwTiling.SetWeightShape(d, h, w);
conv3dBpDwTiling.SetPadding(padFront, padBack, padUp, padDown, padLeft, padRight);
conv3dBpDwTiling.SetStride(strideD, strideH, strideW);
conv3dBpDwTiling.SetDilation(dilationD, dilationH, dilationW);
int ret = conv3dBpDwTiling.GetTiling(tilingData); // 如果 ret = -1, 获取tiling结果失败
```

需要包含的头文件：
```cpp
#include "lib/conv_backprop/conv3d_bp_filter_tiling.h"
```

## 构造函数

### 功能说明

用于创建一个Conv3DBackpropFilter单核Tiling对象。

### 函数原型

- **带参构造函数**，需要传入硬件平台信息，推荐使用这类构造函数来获得更好的兼容性
  - 使用PlatformAscendC类传入信息
    ```cpp
    explicit Conv3dBpFilterTiling(const platform_ascendc::PlatformAscendC& ascendcPlatform)
    ```
  - 使用PlatformInfo传入信息
    ```cpp
    explicit Conv3dBpFilterTiling(const PlatformInfo& platform)
    ```

  当`platform_ascendc::PlatformAscendC`无法在Tiling运行时获取时，需要用户自己构造PlatformInfo结构体，透传给Conv3dBpFilterTiling构造函数。

- **无参构造函数**
  ```cpp
  Conv3dBpFilterTiling()
  ```

- **基类构造函数**
  Conv3dBpFilterTiling继承自基类Conv3dBpFilterTilingBase，其构造函数如下：
  ```cpp
  Conv3dBpFilterTilingBase()
  explicit Conv3dBpFilterTilingBase(const platform_ascendc::PlatformAscendC& ascendcPlatform)
  explicit Conv3dBpFilterTilingBase(const PlatformInfo& platform)
  ```

### 参数说明

**表 15-959 参数说明**

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| ascendcPlatform | 输入 | 传入硬件平台的信息，PlatformAscendC定义请参见构造及析构函数 |
| platform | 输入 | 传入硬件版本以及AI Core中各个硬件单元提供的内存大小。PlatformInfo构造时通过构造及析构函数获取 |

PlatformInfo结构定义如下，socVersion通过GetSocVersion获取并透传，各类硬件存储空间大小通过GetCoreMemSize获取并透传：

```cpp
struct PlatformInfo {
    platform_ascendc::SocVersion socVersion;
    uint64_t l1Size = 0;
    uint64_t l0CSize = 0;
    uint64_t ubSize = 0;
    uint64_t l0ASize = 0;
    uint64_t l0BSize = 0;
};
```

不推荐通过直接填值构造PlatformInfo的方式调用构造函数，例如`PlatformInfo(, 1024, 1024, ..);`

在实现Host侧的Tiling函数时，`platform_ascendc::PlatformAscendC`用于获取一些硬件平台的信息，来支撑Tiling的计算，比如获取硬件平台的核数等信息。PlatformAscendC类提供获取这些平台信息的功能。

和`platform_ascendc::PlatformAscendC`不同的是，PlatformInfo则用于获取芯片版本以及AI Core中各个硬件单元提供的内存大小等只针对单个AI Core的信息。

### 约束说明

无

### 使用样例

- **无参构造函数**
```cpp
Convolution3DBackprop::Conv3dBpFilterTiling tiling;
tiling.SetWeightType(ConvCommonApi::TPosition::GM,
                    ConvCommonApi::ConvFormat::FRACTAL_Z_3D,
                    ConvCommonApi::ConvDtype::FLOAT32);
...
optiling::Conv3DBackpropFilterTilingData tilingData;
int ret = tiling.GetTiling(tilingData); // if ret = -1, gen tiling failed
...
```

- **带参构造函数**
```cpp
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
ConvBackpropApi::Conv3dBpFilterTiling tiling(ascendcPlatform);
tiling.SetWeightType(ConvCommonApi::TPosition::GM,
                    Convolution3DBackprop::ConvFormat::FRACTAL_Z_3D,
                    ConvCommonApi::ConvDtype::FLOAT32);
...
optiling::Conv3DBackpropFilterTilingData tilingData;
int ret = tiling.GetTiling(tilingData); // if ret = -1, gen tiling failed
```
