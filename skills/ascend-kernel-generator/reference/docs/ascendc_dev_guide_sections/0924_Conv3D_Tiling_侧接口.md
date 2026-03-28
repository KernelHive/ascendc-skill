###### Conv3D Tiling 侧接口

## 使用说明

Ascend C提供一组Conv3D Tiling API，方便用户获取Conv3D正向算子Kernel计算时所需的Tiling参数。用户只需要传入Input/Weight/Bias/Output的Position位置、Format格式和DType数据类型及相关参数等信息，调用API接口，即可获取Init中TConv3DApiTiling结构体中的相关参数。

Conv3D Tiling API提供Conv3D单核Tiling接口，用于Conv3D单核计算场景，获取Tiling参数的流程如下：

1. 创建一个单核Tiling对象
2. 设置Input、Weight、Bias、Output的参数类型信息以及Shape信息，如果存在Padding、Stride、Dilation参数，通过SetPadding、SetStride、SetDilation接口进行相关设置
3. 调用GetTiling接口，获取Tiling信息

使用Conv3D Tiling接口获取Tiling参数的样例如下：

```cpp
// 实例化Conv3D Api
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
Conv3dTilingApi::Conv3dTiling conv3dApiTiling(ascendcPlatform);

// 设置输入输出原始规格、单核规格、参数等
conv3dApiTiling.SetGroups(groups);
conv3dApiTiling.SetOrgWeightShape(cout, kd, kh, kw);
conv3dApiTiling.SetOrgInputShape(cin, di, hi, wi);
conv3dApiTiling.SetPadding(padh, padt, padu, padd, padl, padr);
conv3dApiTiling.SetDilation(dilationH, dilationW, dilationD);
conv3dApiTiling.SetStride(strideH, strideW, strideD);
conv3dApiTiling.SetSingleWeightShape(cin, kd, kh, kw);
conv3dApiTiling.SetSingleOutputShape(singleCoreCo, singleCoreDo, singleCoreMo);

// 设置输入输出type
conv3dApiTiling.SetInputType(TPosition::GM, inputFormat, inputDtype);
conv3dApiTiling.SetWeightType(TPosition::GM, weightFormat, weightDtype);
conv3dApiTiling.SetOutputType(TPosition::CO1, outputFormat, outputDtype);
if (biasFlag) {
    conv3dApiTiling.SetBiasType(TPosition::GM, biasFormat, biasDtype);
}

// 调用GetTiling接口获取核内切分策略，如果返回-1代表获取tiling失败
if (conv3dApiTiling.GetTiling(tilingData.conv3ApiTilingData) == -1) {
    return false;
}
```

需要包含的头文件：
```cpp
#include "lib/conv/conv3d/conv3d_tiling.h"
```

## 构造函数

### 功能说明

用于创建一个Conv3D单核Tiling对象。

### 函数原型

```cpp
explicit Conv3dTiling(const platform_ascendc::PlatformAscendC& ascendcPlatform)
explicit Conv3dTilingBase(const PlatformInfo& platform)
```

### 参数说明

**表 参数说明**

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
    uint64_t btSize = 0;
    uint64_t fbSize = 0;
};
```

### 约束说明

无

### 调用示例

```cpp
// 实例化Conv3d Api
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
Conv3dTilingApi::Conv3dTiling conv3dApiTiling(ascendcPlatform);
conv3dApiTiling.SetGroups(groups);
conv3dApiTiling.SetOrgWeightShape(cout, kd, kh, kw);
...
conv3dApiTiling.GetTiling(conv3dCustomTilingData.conv3dApiTilingData);
```
