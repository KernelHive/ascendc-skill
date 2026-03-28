### GlobalLpPool

## 功能
接收一个输入 Tensor，并在同一通道内对其值应用 LpPool 池化操作。这相当于使用核大小等于输入张量空间维度的 LpPool 池化。

## 输入
- **input**：输入 Tensor
  - 数据类型：float16、float

## 属性
- **p**：用于池化输入数据的 Lp 范数的 p 值
  - 数据类型：int32
  - 默认取值：2

## 输出
- **y**：输出 Tensor
  - 数据类型：与输入一致

## 约束与限制
无。

## 支持的 ONNX 版本
Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
