### ReduceSumSquare

## 功能
沿所指定的轴计算输入 Tensor 元素的平方和。如果 `keepdims` 等于 1，得到的 Tensor 的维数与输入的相同。如果 `keepdims` 等于 0，那么生成的 Tensor 会去除被缩减的维度。

## 输入
- **data**：输入 Tensor，数据类型：uint32、uint64、int32、int64、float16、float、double。

## 输出
- **reduced data**：输出 Tensor，数据类型同输入 data 一致。

## 属性
- **axes**：数据类型为 Int 列表；指定计算轴；取值范围：[-r, r-1]，r 是输入数据 data 的维数。
- **keepdims**：数据类型 int，是否保留缩减的维度；默认值为 1（保留）。

## 限制与约束
无

## 支持的 ONNX 版本
Opset v1/v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
