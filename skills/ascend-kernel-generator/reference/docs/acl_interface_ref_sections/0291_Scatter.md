### Scatter

## 功能
根据 `updates` 和 `indices` 来更新 `data` 的值，并把结果返回。

## 输入
- **data**：输入 Tensor，数据类型支持 float16、float、int32。
- **indices**：输入 Tensor，数据类型支持 int32、int64。shape 与 `data` 一致。
- **updates**：输入 Tensor，数据类型同 `data`。shape 与 `data` 一致。

## 输出
- **y**：输出 Tensor，与输入 `data` 的 shape、数据类型一致。

## 属性
- **axis**：int，默认是 0，表示沿 `axis` 取数据。

## 支持的 ONNX 版本
Opset v9/v10/v11
