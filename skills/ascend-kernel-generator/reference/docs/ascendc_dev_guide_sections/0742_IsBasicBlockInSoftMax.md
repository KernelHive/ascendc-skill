###### IsBasicBlockInSoftMax

## 功能说明

用于判断 SoftMaxTiling 结构是否符合基本块特征。

## 函数原型

### AscendC::optiling 命名空间下的计算接口

```cpp
bool IsBasicBlockInSoftMax(optiling::SoftMaxTiling& tiling, const uint32_t dataTypeSize = 2)
```

### AscendC 命名空间下的计算接口

```cpp
bool IsBasicBlockInSoftMax(SoftMaxTiling& tiling, const uint32_t dataTypeSize = 2)
```

## 参数说明

| 参数名 | 输入/输出 | 功能描述 |
|--------|-----------|----------|
| tiling | 输入 | 待判断的 SoftMaxTiling 结构，支持 `optiling::SoftMaxTiling` 形式入参和 `SoftMaxTiling` 形式入参 |
| dataTypeSize | 输入 | 参与计算的 srcTensor 的数据类型大小，例如 half=2 |

## 返回值说明

- 返回 `true` 表示 SoftMaxTiling 结构满足基本块 Tiling 特征
- 返回 `false` 表示 SoftMaxTiling 结构不满足基本块 Tiling 特征

## 约束说明

无
