#### ShapeInfo

## 功能说明

ShapeInfo 用来存放 LocalTensor 或 GlobalTensor 的 shape 信息。

## 定义原型

### ShapeInfo 结构定义

```cpp
struct ShapeInfo {
public:
    __aicore__ inline ShapeInfo();
    __aicore__ inline ShapeInfo(const uint8_t inputShapeDim, const uint32_t inputShape[],
                                const uint8_t inputOriginalShapeDim, const uint32_t inputOriginalShape[], 
                                const DataFormat inputFormat);
    __aicore__ inline ShapeInfo(const uint8_t inputShapeDim, const uint32_t inputShape[], 
                                const DataFormat inputFormat);
    __aicore__ inline ShapeInfo(const uint8_t inputShapeDim, const uint32_t inputShape[]);
    
    uint8_t shapeDim;
    uint8_t originalShapeDim;
    uint32_t shape[K_MAX_DIM];
    uint32_t originalShape[K_MAX_DIM];
    DataFormat dataFormat;
};
```

### 获取 Shape 中所有 dim 的累乘结果

```cpp
__aicore__ inline int GetShapeSize(const ShapeInfo& shapeInfo)
```

## 函数说明

### ShapeInfo 结构参数说明

| 参数名称 | 描述 |
|----------|------|
| shapeDim | 现有的 shape 维度 |
| shape | 现有的 shape |
| originalShapeDim | 原始的 shape 维度 |
| originalShape | 原始的 shape |
| dataFormat | 数据排布格式，DataFormat 类型，定义如下：<br>`enum class DataFormat : uint8_t {`<br>&nbsp;&nbsp;`ND = 0,`<br>&nbsp;&nbsp;`NZ,`<br>&nbsp;&nbsp;`NCHW,`<br>&nbsp;&nbsp;`NC1HWC0,`<br>&nbsp;&nbsp;`NHWC,`<br>`};` |

### GetShapeSize 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| shapeInfo | 输入 | ShapeInfo 类型，LocalTensor 或 GlobalTensor 的 shape 信息 |
