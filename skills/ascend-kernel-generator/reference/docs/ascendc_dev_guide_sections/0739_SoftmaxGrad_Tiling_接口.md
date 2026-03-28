###### SoftmaxGrad Tiling 接口

## 功能说明

用于获取 SoftmaxGrad Tiling 参数。

## 函数原型

### 获取临时空间大小的接口

```cpp
uint32_t GetSoftMaxGradMaxTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize, const bool isFront, const bool isReuseSource)
uint32_t GetSoftMaxGradMinTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize, const bool isFront, const bool isReuseSource)
```

### Tiling 计算接口

#### AscendC::optiling 命名空间下的计算接口

```cpp
void SoftMaxGradTilingFunc(const ge::Shape& srcShape, const uint32_t dataTypeSize, const uint32_t localWorkSpaceSize, optiling::SoftMaxTiling& softmaxGradTiling, const bool isFront = false)
```

#### AscendC 命名空间下的计算接口

```cpp
void SoftMaxGradTilingFunc(const ge::Shape& srcShape, const uint32_t dataTypeSize, const uint32_t localWorkSpaceSize, SoftMaxTiling& softmaxGradTiling, const bool isFront = false)
```

## 参数说明

### GetSoftMaxGradMaxTmpSize/GetSoftMaxGradMinTmpSize 接口参数

| 参数 | 输入/输出 | 功能说明 |
|------|-----------|----------|
| srcShape | 输入 | 输入 srcTensor 的 shape 信息 |
| dataTypeSize | 输入 | 计算的数据类型，比如 half=2 |
| isFront | 输入 | 是否只计算，和 kernel 侧的 SoftmaxGrad 接口一致，默认 false |
| isReuseSource | 输入 | 与 kernel 侧接口配置保持一致 |

### SoftMaxGradTilingFunc 接口参数

| 参数 | 输入/输出 | 功能说明 |
|------|-----------|----------|
| srcShape | 输入 | 输入 srcTensor 的 shape 信息 |
| dataTypeSize | 输入 | 计算的数据类型，比如 half=2 |
| localWorkSpaceSize | 输入 | 剩余的可供 SoftmaxGrad 接口计算的临时空间大小，单位为 Byte。localWorkSpaceSize 的取值必须大于 GetSoftMaxGradMinTmpSize 接口返回的计算所需的最小临时空间大小 |
| isFront | 输入 | 是否只计算，和 kernel 侧的 SoftmaxGrad 接口一致，默认 false |
| softmaxGradTiling | 输出 | 输出 SoftmaxGrad 接口所需的 tiling 信息，支持 optiling::SoftMaxTiling 形式入参和 SoftMaxTiling 形式入参 |

## 返回值说明

- `GetSoftMaxGradMinTmpSize` 返回 SoftmaxGrad 接口能完成计算所需最小临时空间大小，单位为 Byte
- `GetSoftMaxGradMaxTmpSize` 返回 SoftmaxGrad 接口能完成计算所需最大临时空间大小，单位为 Byte

## 约束说明

无
