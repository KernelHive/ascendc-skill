###### SoftmaxFlash Tiling 接口

## 功能说明

> **注意**：该接口后续即将废弃，新开发内容不要使用该接口。

用于获取 SoftmaxFlash Tiling 参数。

## 函数原型

### 获取临时空间大小接口

获取 Kernel 接口计算所需最小/最大临时空间的接口：

```cpp
uint32_t GetSoftMaxFlashMaxTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize, const bool isUpdate, const bool isReuseSource)
uint32_t GetSoftMaxFlashMinTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize, const bool isUpdate, const bool isReuseSource)
```

### Tiling 计算接口

#### AscendC::optiling 命名空间下的计算接口

```cpp
void SoftMaxFlashTilingFunc(const ge::Shape& srcShape, const uint32_t dataTypeSize, const uint32_t localWorkSpaceSize, optiling::SoftMaxTiling& softmaxFlashTiling, const bool isUpdate = false)
```

#### AscendC 命名空间下的计算接口

```cpp
void SoftMaxFlashTilingFunc(const ge::Shape& srcShape, const uint32_t dataTypeSize, const uint32_t localWorkSpaceSize, SoftMaxTiling& softmaxFlashTiling, const bool isUpdate = false)
```

## 参数说明

### GetSoftMaxFlashMaxTmpSize/GetSoftMaxFlashMinTmpSize 接口参数

| 参数 | 输入/输出 | 功能说明 |
|------|-----------|----------|
| srcShape | 输入 | 输入 srcTensor 的 shape 信息 |
| dataTypeSize | 输入 | 参与计算的 maxTensor 和 sumTensor 的数据类型，比如 half=2 |
| isUpdate | 输入 | 是否使能刷新功能，和 kernel 侧 SoftmaxFlash 接口一致，默认 false |
| isReuseSource | 输入 | 与 kernel 侧接口配置保持一致 |

### SoftMaxFlashTilingFunc 接口参数

| 参数 | 输入/输出 | 功能说明 |
|------|-----------|----------|
| srcShape | 输入 | 输入 srcTensor 的 shape 信息 |
| dataTypeSize | 输入 | 参与计算的 maxTensor 和 sumTensor 的数据类型，比如 half=2 |
| localWorkSpaceSize | 输入 | 剩余的可供 SoftmaxFlash 接口计算的空间大小，单位为 Byte。<br>localWorkSpaceSize 的取值必须大于 GetSoftMaxFlashMinTmpSize 接口返回的计算所需的最小临时空间大小 |
| isUpdate | 输入 | 是否使能刷新功能，和 kernel 侧 SoftmaxFlash 接口一致，默认 false |
| softmaxFlashTiling | 输出 | 输出 SoftmaxFlash 接口所需的 tiling 信息，支持 optiling::SoftMaxTiling 形式入参和 SoftMaxTiling 形式入参 |

## 返回值说明

- `GetSoftMaxFlashMaxTmpSize` 返回 SoftmaxFlash 接口能完成计算所需最大临时空间大小，单位为 Byte
- `GetSoftMaxFlashMinTmpSize` 返回 SoftmaxFlash 接口能完成计算所需最小临时空间大小，单位为 Byte

## 约束说明

无
