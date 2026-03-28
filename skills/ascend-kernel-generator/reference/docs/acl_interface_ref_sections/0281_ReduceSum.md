### ReduceSum

## 功能
计算输入 Tensor 指定维度的元素的和。如果 `keepdims` 等于 1，得到的 Tensor 的维数与输入的相同。如果 `keepdims` 等于 0，那么生成的 Tensor 会去除被缩减的维度。

## 输入
- **data**：输入 Tensor，数据类型支持 float16、float。

## 输出
- **reduced data**：输出 Tensor，和输入 data 的数据类型一致。

## 属性
- **axes**：数据类型为 Int 列表；指定计算轴；取值范围：`[-r, r-1]`，r 是输入数据 data 的维数。
- **keepdims**（默认值为 "1"）：数据类型为 int，是否保留缩减的维度；默认值为 1（保留）。

## 限制与约束
`axes` 为空时，当前输出不做全维度规约。此时建议：
- 修改 ONNX 算子的 `axes` 为所有轴；或者
- 在 PyTorch 导出 ONNX 图前，在 PyTorch 模型中使用 `amax` 来对所有轴规约（例如：`x.amax(dim=[0, 1, 2])`）。

## 支持的 ONNX 版本
Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
