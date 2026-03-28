### RandomNormalLike

## 功能
根据正态分布生成随机数矩阵。

## 输入
**x**：输入 Tensor，数据类型支持：
- float64
- int8
- int16
- int32
- int64
- uint8
- uint16
- uint32
- uint64
- float16
- float

## 属性
- **dtype**：数据类型为 int，指定输出 Tensor 的数据类型。
- **mean**：数据类型为 float，默认值为 0.0，正态分布的均值。
- **scale**：数据类型为 float，默认值为 1.0，正态分布的标准差。
- **seed**：数据类型为 float，随机数种子。

## 输出
**y**：输出 Tensor，shape、数据类型和输入 x 一致。

## 约束与限制
无。

## 支持的 ONNX 版本
Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18。
