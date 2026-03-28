###### Conv3DBackpropInput Tiling 侧接口

## 使用说明

Ascend C提供一组Conv3DBackpropInput Tiling API，方便用户获取Conv3DBackpropInput Kernel计算时所需的Tiling参数。用户只需要传入Input/GradOutput/Weight的Position位置、Format格式和DType数据类型及相关参数等信息，调用API接口，即可获取Init中TConv3DBackpropInputTiling结构体中的相关参数。

Conv3DBackpropInput Tiling API提供一个GetTiling接口获取Tiling参数，获取Tiling参数的流程如下：

1. 创建一个单核Tiling对象
2. 设置Input、GradOutput、Weight的参数类型信息以及Shape信息，如果存在Padding、Stride参数，通过SetPadding、SetStride接口设置
3. 调用GetTiling接口，获取Tiling信息

使用Conv3DBackpropInput Tiling接口获取Tiling参数的样例如下：

```cpp
#include "tiling/conv_backprop/conv3d_bp_input_tiling.h"

optiling::Conv3DBackpropInputTilingData tilingData;
auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
ConvBackpropApi::Conv3DBpInputTiling conv3DBpDxTiling(*ascendcPlatform);
conv3DBpDxTiling.SetWeightType(Convolution3DBackprop::TPosition::GM,
    Convolution3DBackprop::ConvFormat::FRACTAL_Z_3D,
    Convolution3DBackprop::ConvDtype::FLOAT32);
conv3DBpDxTiling.SetGradOutputType(Convolution3DBackprop::TPosition::GM,
    Convolution3DBackprop::ConvFormat::NDC1HWC0,
    Convolution3DBackprop::ConvDtype::FLOAT16);
conv3DBpDxTiling.SetInputType(Convolution3DBackprop::TPosition::CO1,
    Convolution3DBackprop::ConvFormat::NDC1HWC0,
    Convolution3DBackprop::ConvDtype::FLOAT16);
conv3DBpDxTiling.SetInputShape(orgN, orgCi, orgDi, orgHi, orgWi);
conv3DBpDxTiling.SetGradOutputShape(orgCo, orgDo, orgHo, orgWo);
conv3DBpDxTiling.SetWeightShape(orgKd, orgKh, orgKw);
conv3DBpDxTiling.SetPadding(padFront, padBack, padUp, padDown, padLeft, padRight);
conv3DBpDxTiling.SetStride(strideD, strideH, strideW);
conv3DBpDxTiling.SetDilation(dilationD, dilationH, dilationW);
int ret = conv3DBpDxTiling.GetTiling(tilingData); // if ret = -1, get tiling failed
```

需要包含的头文件：
```cpp
#include "lib/conv_backprop/conv3d_bp_input_tiling.h"
```

## 构造函数

### 功能说明

用于创建一个Conv3DBackpropInput单核Tiling对象。

### 函数原型

- **带参构造函数**，需要传入硬件平台信息，推荐使用这类构造函数来获得更好的兼容性
  - 使用PlatformAscendC类传入信息
    ```cpp
    explicit Conv3DBpInputTiling(const platform_ascendc::PlatformAscendC& ascendcPlatform)
    ```
  - 使用PlatformInfo传入信息
    ```cpp
    explicit Conv3DBpInputTiling(const PlatformInfo& platform)
    ```

- **无参构造函数**
  ```cpp
  Conv3DBpInputTiling()
  ```

- **基类构造函数**
  Conv3DBpInputTiling继承自基类Conv3DBpInputTilingBase，其构造函数如下：
  ```cpp
  Conv3DBpInputTilingBase()
  explicit Conv3DBpInputTilingBase(const platform_ascendc::PlatformAscendC& ascendcPlatform)
  explicit Conv3DBpInputTilingBase(const PlatformInfo& platform)
  ```

### 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| ascendcPlatform | 输入 | 传入硬件平台的信息，PlatformAscendC定义请参见构造及析构函数 |
| platform | 输入 | 传入硬件版本以及AI Core中各个硬件单元提供的内存大小。PlatformInfo构造时通过构造及析构函数获取 |

PlatformInfo结构定义如下：
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

### 约束说明

无

### 调用示例

- **无参构造函数**
```cpp
ConvBackpropApi::Conv3DBpInputTiling tiling;
tiling.SetWeightType(ConvCommonApi::TPosition::GM,
    ConvCommonApi::ConvFormat::FRACTAL_Z_3D,
    ConvCommonApi::ConvDtype::FLOAT16);
...
optiling::Conv3DBackpropInputTilingData tilingData;
int ret = tiling.GetTiling(tilingData); // if ret = -1, gen tiling failed
...
```

- **带参构造函数**
```cpp
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
ConvBackpropApi::Conv3DBpInputTiling tiling(ascendcPlatform);
tiling.SetWeightType(ConvCommonApi::TPosition::GM,
    ConvCommonApi::ConvFormat::FRACTAL_Z_3D,
    ConvCommonApi::ConvDtype::FLOAT16);
...
optiling::Conv3DBackpropInputTilingData tilingData;
int ret = tiling.GetTiling(tilingData); // if ret = -1, gen tiling failed
```
