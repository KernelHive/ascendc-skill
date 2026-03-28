### Gather

## 功能
对输入 Tensor `x` 中指定的维度 `dim` 进行数据切片。

## 输入
- **x**：输入 Tensor，数据类型：float16、float、int32、int64、int8、int16、uint8、uint16、uint32、uint64、bool。
- **indices**：输入 Tensor，数据类型：int32、int64。

## 属性
- **axis**：数据类型支持 int，指定 gather 的轴，取值范围为 `[-r, r-1]`（`r` 表示输入数据的秩）。

## 输出
- **y**：输出 Tensor，数据类型和输入 `x` 类型一致。

## 约束
不支持 `indices` 为负值的索引。

## 支持的 ONNX 版本
Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
