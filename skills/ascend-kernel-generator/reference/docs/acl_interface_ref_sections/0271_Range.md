### Range

## 功能
从 `start` 起始到 `end` 结束按照 `step` 的间隔取值，生成连续序列的 Tensor。

## 输入
- **start**：scalar，数据类型支持 float16、float、int32、int64。
- **end**：scalar，数据类型和 `start` 一致。
- **step**：scalar，数据类型支持 float16、float、int32、int64。

## 输出
- **y**：输出 Tensor，数据类型和 `start` 一致。

## 约束与限制
无。

## 支持的 ONNX 版本
Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18。
