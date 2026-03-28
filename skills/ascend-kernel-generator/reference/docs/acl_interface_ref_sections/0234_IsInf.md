### IsInf

## 功能
将输入 Tensor 的无穷大元素映射为 `true`，将其他值映射为 `false`。

## 输入
- **x**：输入 Tensor，数据类型支持 `float32`、`double`。

## 属性
- **detect_negative**（可选）：数据类型为 `int`，是否将负无穷大映射为 `true`。默认为 `1`，以便负无穷大推导为真。如果负无穷大应映射为 `false`，则将此属性设置为 `0`。
- **detect_positive**（可选）：数据类型为 `int`，是否将正无穷大映射为 `true`。默认为 `1`，以便正无穷大推导为真。如果正无穷大应映射为 `false`，则将此属性设置为 `0`。

## 输出
- **y**：输出 Tensor，shape 与输入 `x` 一致，数据类型：`bool`。

## 约束与限制
无。

## 支持的 ONNX 版本
Opset v10/v11/v12/v13/v14/v15/v16/v17/v18
