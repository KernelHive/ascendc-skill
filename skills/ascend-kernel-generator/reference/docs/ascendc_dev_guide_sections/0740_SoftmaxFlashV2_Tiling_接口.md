###### SoftmaxFlashV2 Tiling 接口

## 功能说明

用于获取 SoftmaxFlashV2 接口所需的 Tiling 参数。

## 函数原型

### 获取 Kernel 接口计算所需最小/最大临时空间的接口

```cpp
uint32_t GetSoftMaxFlashV2MinTmpSize(const ge::Shape& srcShape, 
                                     const uint32_t dataTypeSize1,
                                     const uint32_t dataTypeSize2, 
                                     const bool isUpdate, 
                                     const bool isBasicBlock = false, 
                                     const bool isFlashOutputBrc = false)

uint32_t GetSoftMaxFlashV2MaxTmpSize(const ge::Shape& srcShape, 
                                     const uint32_t dataTypeSize1,
                                     const uint32_t dataTypeSize2, 
                                     const bool isUpdate, 
                                     const bool isBasicBlock = false, 
                                     const bool isFlashOutputBrc = false)
```

### Tiling 计算接口

#### AscendC::optiling 命名空间下的计算接口

```cpp
void SoftMaxFlashV2TilingFunc(const ge::Shape& srcShape, 
                              const uint32_t dataTypeSize1,
                              const uint32_t dataTypeSize2, 
                              const uint32_t localWorkSpaceSize,
                              optiling::SoftMaxTiling& softmaxFlashTiling,
                              const bool isUpdate, 
                              const bool isBasicBlock = false, 
                              const bool isFlashOutputBrc = false)
```

#### AscendC 命名空间下的计算接口

```cpp
void SoftMaxFlashV2TilingFunc(const ge::Shape& srcShape, 
                              const uint32_t dataTypeSize1,
                              const uint32_t dataTypeSize2, 
                              const uint32_t localWorkSpaceSize,
                              SoftMaxTiling& softmaxFlashTiling,
                              const bool isUpdate, 
                              const bool isBasicBlock = false, 
                              const bool isFlashOutputBrc = false)
```

## 参数说明

### GetSoftMaxFlashV2MinTmpSize/GetSoftMaxFlashV2MaxTmpSize 接口参数列表

| 参数名 | 输入/输出 | 功能说明 |
|--------|-----------|----------|
| srcShape | 输入 | 输入 srcTensor 的 shape 信息 |
| dataTypeSize1 | 输入 | 计算的源数据的数据类型大小，比如 half=2 |
| dataTypeSize2 | 输入 | 参与计算的 expSumTensor 和 maxTensor 的数据类型大小，比如 half=2 |
| isUpdate | 输入 | 是否使能刷新功能，和 kernel 侧 SoftmaxFlashV2 接口一致 |
| isBasicBlock | 输入 | 是否要使能基本块计算。isBasicBlock 参数可以通过 isBasicBlockInSoftmax 接口获取，与 kernel 侧接口的模板参数保持一致，默认 false。注意，若 kernel 侧 API 使能模板参数 SoftmaxConfig，即 shape 常量化场景，isBasicBlock 参数必须通过接口 isBasicBlockInSoftmax 获取 |
| isFlashOutputBrc | 输入 | 是否使能输出 shape 的非拓展模式。非拓展模式为不对输出数据做 Broadcast，输出 shape 为 (m, 1)。参数取值如下：<br>• false：不使能非拓展模式，默认值。输出为 float 数据类型时，shape 为 (m, 8)；输出为 half 数据类型时，shape 为 (m, 16)<br>• true：使能非拓展模式，输出的 shape 均为 (m, 1)。该参数取值为 true 时，kernel 接口的模板参数 SoftmaxConfig 中的 mode 必须配置为 SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC |

### SoftMaxFlashV2TilingFunc 接口参数列表

| 参数名 | 输入/输出 | 功能说明 |
|--------|-----------|----------|
| srcShape | 输入 | 输入 srcTensor 的 shape 信息 |
| dataTypeSize1 | 输入 | 计算的源数据的数据类型，比如 half=2 |
| dataTypeSize2 | 输入 | 参与计算的 maxTensor 和 sumTensor 的数据类型，比如 half=2 |
| localWorkSpaceSize | 输入 | 剩余的可供 SoftmaxFlashV2 接口计算的空间大小。localWorkSpaceSize 的取值必须大于 GetSoftMaxFlashV2MinTmpSize 接口返回的计算所需的最小临时空间大小 |
| isUpdate | 输入 | 是否使能刷新功能，和 kernel 侧 SoftmaxFlashV2 接口一致 |
| isBasicBlock | 输入 | 是否要使能基本块计算。isBasicBlock 参数可以通过 isBasicBlockInSoftmax 接口获取，与 kernel 侧接口的模板参数保持一致，默认 false。注意，若 kernel 侧 API 使能模板参数 SoftmaxConfig，即 shape 常量化场景，isBasicBlock 参数必须通过接口 isBasicBlockInSoftmax 获取 |
| isFlashOutputBrc | 输入 | 是否使能输出 shape 的非拓展模式。非拓展模式为不对输出数据做 Broadcast，输出 shape 为 (m, 1)。参数取值如下：<br>• false：不使能非拓展模式，默认值。输出为 float 数据类型时，shape 为 (m, 8)；输出为 half 数据类型时，shape 为 (m, 16)<br>• true：使能非拓展模式，输出的 shape 均为 (m, 1)。该参数取值为 true 时，kernel 接口的模板参数 SoftmaxConfig 中的 mode 必须配置为 SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC |
| softmaxFlashTiling | 输出 | 输出 SoftmaxFlashV2 接口所需的 tiling 信息，支持 optiling::SoftMaxTiling 形式入参和 SoftMaxTiling 形式入参 |

## 返回值说明

- `GetSoftMaxFlashV2MinTmpSize` 返回 SoftmaxFlashV2 接口能完成计算所需最小临时空间大小，单位为 Byte
- `GetSoftMaxFlashV2MaxTmpSize` 返回 SoftmaxFlashV2 接口能完成计算所需最大临时空间大小，单位为 Byte

## 约束说明

无
