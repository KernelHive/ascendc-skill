### Dropout

## 功能

Dropout 采用输入浮点张量、可选输入比率（浮点标量）和可选输入 training_mode（bool 标量）。它产生两个张量输出 output（浮点张量）和 mask（可选）。

如果 training_mode 为 true，则输出 Y 将为随机 dropout；请注意，此 Dropout 按以下公式缩放掩码输入数据，因此要将训练后的模型转换为推理模式，用户可以简单地不传递 input 或将其设置为 false。

## 输入

- **data**：输入 Tensor，数据类型：float16、float
- **ratio**（可选）：随机 dropout 的比率，值为 [0, 1），默认为 0.5，数据类型：float16、float
- **training_mode**（可选）：如果设置为 true，则表示 dropout 正在用于训练，除非明确指定，否则它是 false。如果为 false，则忽略 ratio，并且该操作模拟推理模式，在该模式下，不会从输入数据中删除任何内容，如果请求 mask 作为输出，它将包含所有 1，数据类型：bool

## 输出

- **output**：输出 Tensor，数据类型：和输入 data 一致
- **mask**（可选）：输出 Tensor，输出掩码，数据类型：bool

## 约束与限制

无

## 支持的 ONNX 版本

Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
