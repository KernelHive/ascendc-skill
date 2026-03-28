### ReduceMin

## 功能
沿着指定的轴计算输入 Tensor 元素的最小值。如果 `keepdims` 等于 1，那么生成的 Tensor 与输入 Tensor 具有相同的维数。如果 `keepdims` 等于 0，那么生成的 Tensor 会去除被缩减的维度。

## 输入
- **data**：输入 Tensor
  - 数据类型：uint8、int32、float16、float
  - 数据格式支持：ND

## 输出
- **reduced data**：输出 Tensor
  - 数据类型：与输入 data 一致

## 属性
- **keepdims**：数据类型为 int
  - 是否保留缩减的维度
  - 默认值：1（保留）
- **axes**（可选）：数据类型为 Int64
  - 指定计算轴
  - 取值范围：[-r, r-1]，r 是输入数据 data 的维数

## 限制与约束
`axes` 为空时，当前输出不做全维度规约。此时建议：
- 修改 ONNX 算子的 `axes` 为所有轴；或者
- 在 torch 导出 ONNX 图前，在 torch 模型中使用 `amax` 来对所有轴规约（例如：`x.amax(dim=[0, 1, 2])`）

## 支持的 ONNX 版本
Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
