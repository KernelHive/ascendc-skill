### AscendQuant

## 功能
对输入 `x` 进行量化操作，`scale` 和 `offset` 的 size 需要是 `x` 的最后一维或 1。

## 输入
- **x**：输入 Tensor
  - 数据类型支持：float16、float
  - 数据格式支持：ND

## 属性
- **offset**
  - 数据类型：float
- **scale**
  - 数据类型：float
- **sqrt_mode**
  - 数据类型：bool
- **round_mode**
  - 数据类型：string

## 输出
- **y**：输出 Tensor
  - 数据类型支持：int8
  - 数据格式支持：ND

## 约束与限制
Atlas 推理系列产品不支持 `scale`、`offset` 及输入 `x` 为 bfloat16。

## 支持的 ONNX 版本
Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18/v19/v20/v21/v22
