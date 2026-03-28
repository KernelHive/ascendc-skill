### Compress

## 功能
沿给定轴从输入张量中选择切片，其中每个轴索引的 `condition` 计算结果为 `True`。

## 输入
- **input**：秩 `r >= 1` 的 Tensor，数据类型支持：
  - uint8、uint16、uint32、uint64
  - int8、int16、int32、int64
  - float16、float
- **condition**：一维 Tensor，用于指定切片和需要选择的元素，数据类型支持 `bool`。

## 属性
- **axis**（可选）：数据类型为 `int`，进行切片的轴。如果未提供 `axis`，则在选择元素之前将 `input` 展平。取值范围为 `[-r, r-1]`，其中 `r` 为输入 `input` 的维数。

## 输出
- **output**：输出 Tensor。如果指定了 `axis`，则为秩 `r` 的 Tensor，否则是秩为 1 的 Tensor。数据类型与输入 `input` 一致。

## 约束与限制
无。

## 支持的 ONNX 版本
Opset v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
