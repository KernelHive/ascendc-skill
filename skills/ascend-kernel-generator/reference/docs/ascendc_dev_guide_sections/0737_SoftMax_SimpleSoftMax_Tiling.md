###### SoftMax/SimpleSoftMax Tiling

## 功能说明

用于获取 SoftMax/SimpleSoftMax Tiling 参数。

## 函数原型

### 获取临时空间大小接口

```cpp
uint32_t GetSoftMaxMaxTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize, const bool isReuseSource)
uint32_t GetSoftMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize, const bool isReuseSource)
```

### Tiling 计算接口

```cpp
// AscendC::optiling 命名空间下的计算接口
void SoftMaxTilingFunc(const ge::Shape& srcShape, const uint32_t dataTypeSize, const uint32_t localWorkSpaceSize, optiling::SoftMaxTiling& softmaxTiling)

// AscendC 命名空间下的计算接口
void SoftMaxTilingFunc(const ge::Shape& srcShape, const uint32_t dataTypeSize, const uint32_t localWorkSpaceSize, SoftMaxTiling& softmaxTiling)
```

## 参数说明

### GetSoftMaxMaxTmpSize/GetSoftMaxMinTmpSize 接口参数

| 参数名 | 输入/输出 | 功能说明 |
|--------|-----------|----------|
| srcShape | 输入 | 输入 srcTensor 的 shape 信息 |
| dataTypeSize | 输入 | 参与计算的 max 和 sum 的数据类型，比如 half=2 |
| isReuseSource | 输入 | 与 kernel 侧接口配置保持一致 |

### SoftMaxTilingFunc 接口参数

| 参数名 | 输入/输出 | 功能说明 |
|--------|-----------|----------|
| srcShape | 输入 | 输入 srcTensor 的 shape 信息 |
| dataTypeSize | 输入 | 参与计算的 max 和 sum 的数据类型，比如 half=2 |
| localWorkSpaceSize | 输入 | 剩余的可供 SoftMax 接口计算的空间大小，单位为 Byte。<br>localWorkSpaceSize 的取值必须大于 GetSoftMaxMinTmpSize 接口返回的计算所需的最小临时空间大小 |
| softmaxTiling | 输出 | 输出 SoftMax 接口所需的 tiling 信息，支持 optiling::SoftMaxTiling 形式入参和 SoftMaxTiling 形式入参 |

## 返回值说明

- `GetSoftMaxMaxTmpSize` 返回 SoftMax/SimpleSoftMax 接口能完成计算所需最大临时空间大小，单位为 Byte
- `GetSoftMaxMinTmpSize` 返回 SoftMax/SimpleSoftMax 接口能完成计算所需最小临时空间大小，单位为 Byte

## 约束说明

无
