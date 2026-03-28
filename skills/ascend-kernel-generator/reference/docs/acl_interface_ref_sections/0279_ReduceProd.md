### ReduceProd

## 功能
计算输入 Tensor 的元素沿指定轴的乘积。如果 `keepdims` 等于 1，得到的 Tensor 维数与输入 Tensor 相同。如果 `keepdims` 等于 0，那么生成的 Tensor 会去除被缩减的维度。

## 输入
- **data**：输入 Tensor，数据类型为：int8、uint8、int32、float、float16、int64。

## 输出
- **reduced data**：输出 Tensor，数据类型与输入 `data` 一致。

## 属性
- **keepdims**（默认值为 "1"）：数据类型为 int；是否保留缩减的维度；默认为 1（保留）。
- **axes**（可选）：数据类型为 Int64；指定计算轴；取值范围：[-r, r-1]，其中 r 是输入数据 `data` 的维数。

## 限制与约束
无

## 支持的 ONNX 版本
Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
