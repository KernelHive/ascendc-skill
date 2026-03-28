### Shrink

## 功能

对输入张量进行非线性变换，根据输入值 `data` 与阈值 `lambda` 的关系，对输入通过偏移量 `bias` 进行缩放和偏移处理。

## 输入

- **data**：输入 Tensor，数据类型是 float16、float。

## 输出

- **y**：输出 Tensor，和输入 `data` 的 shape、数据类型一致。

## 属性

- **bias**：float，默认是 0.0。
- **lambda**：float，默认是 0.5。

## 支持的 ONNX 版本

Opset v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
