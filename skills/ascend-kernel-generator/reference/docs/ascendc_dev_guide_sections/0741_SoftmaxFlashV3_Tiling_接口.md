###### SoftmaxFlashV3 Tiling 接口

## 功能说明

用于获取 SoftmaxFlashV3 接口所需的 Tiling 参数。

## 函数原型

### 获取 Kernel 接口计算所需最小/最大临时空间的接口

```cpp
void GetSoftMaxFlashV3MaxMinTmpSize(
    const ge::Shape& srcShape,
    const uint32_t dataTypeSize1,
    const uint32_t dataTypeSize2,
    uint32_t& maxValue,
    uint32_t& minValue,
    const bool isUpdate,
    const bool isBasicBlock = false
)
```

### Tiling 计算接口

#### AscendC::optiling 命名空间下的计算接口

```cpp
void SoftMaxFlashV3TilingFunc(
    const ge::Shape& srcShape,
    const uint32_t dataTypeSize1,
    const uint32_t dataTypeSize2,
    const uint32_t localWorkSpaceSize,
    optiling::SoftMaxTiling& softmaxFlashV3Tiling,
    const bool isUpdate,
    const bool isBasicBlock = false
)
```

#### AscendC 命名空间下的计算接口

```cpp
void SoftMaxFlashV3TilingFunc(
    const ge::Shape& srcShape,
    const uint32_t dataTypeSize1,
    const uint32_t dataTypeSize2,
    const uint32_t localWorkSpaceSize,
    SoftMaxTiling& softmaxFlashV3Tiling,
    const bool isUpdate,
    const bool isBasicBlock = false
)
```

## 参数说明

### GetSoftMaxFlashV3MaxMinTmpSize 接口参数列表

| 参数名 | 输入/输出 | 功能说明 |
|--------|-----------|----------|
| srcShape | 输入 | 输入 srcTensor 的 shape 信息。 |
| dataTypeSize1 | 输入 | 输入 srcTensor 的数据类型大小，即对应 SoftMaxFlashV3 Kernel 函数中模板参数 T 的数据类型大小。当前模板参数 T 仅支持 half 类型，故此参数只支持取值为 2。 |
| dataTypeSize2 | 输入 | 输入 inMeanTensor、inExpSumTensor、inMaxTensor 的数据类型大小，即对应 SoftMaxFlashV3 Kernel 函数中模板参数 U 的数据类型大小。当前模板参数 U 仅支持 float 类型，故此参数只支持取值为 4。 |
| maxValue | 输出 | SoftMaxFlashV3 接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel 侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。最大空间大小为 0 表示计算不需要临时空间。<br>**说明**：maxValue 仅作为参考值，有可能大于 Unified Buffer 剩余空间的大小，该场景下，开发者需要根据 Unified Buffer 剩余空间的大小来选取合适的临时空间大小。 |
| minValue | 输出 | SoftMaxFlashV3 接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。最小空间大小为 0 表示计算不需要临时空间。 |
| isUpdate | 输入 | 是否使能 SoftMaxFlashV3 update 为 true 的公式计算。该参数取值与 SoftmaxFlashV3 Kernel 接口的模板参数 isUpdate 保持一致。 |
| isBasicBlock | 输入 | 预留参数，暂未启用，必须使用默认值 false。 |

### SoftMaxFlashV3TilingFunc 接口参数列表

| 参数名 | 输入/输出 | 功能说明 |
|--------|-----------|----------|
| srcShape | 输入 | 输入 srcTensor 的 shape 信息。 |
| dataTypeSize1 | 输入 | 输入 srcTensor 的数据类型大小，即对应 SoftMaxFlashV3 Kernel 函数中模板参数 T 的数据类型大小。当前模板参数 T 仅支持 half 类型，故此参数只支持取值为 2。 |
| dataTypeSize2 | 输入 | 输入 inMeanTensor、inExpSumTensor、inMaxTensor 的数据类型大小，即对应 SoftMaxFlashV3 Kernel 函数中模板参数 U 的数据类型大小。当前模板参数 U 仅支持 float 类型，故此参数只支持取值为 4。 |
| localWorkSpaceSize | 输入 | 剩余的可供 SoftmaxFlashV3 接口计算的空间大小。localWorkSpaceSize 的取值必须大于 GetSoftMaxFlashV3MaxMinTmpSize 接口返回的计算所需的最小临时空间大小。 |
| isUpdate | 输入 | 是否使能 SoftMaxFlashV3 update 为 true 的公式计算。与 SoftmaxFlashV3 Kernel 接口的模板参数 isUpdate 保持一致。 |
| isBasicBlock | 输入 | 预留参数，暂未启用，必须使用默认值 false。 |
| softmaxFlashV3Tiling | 输出 | 输出 SoftMaxFlashV3 接口所需的 Tiling 信息，支持 optiling::SoftMaxTiling 形式入参和 SoftMaxTiling 形式入参。 |

## 返回值说明

无

## 约束说明

无
