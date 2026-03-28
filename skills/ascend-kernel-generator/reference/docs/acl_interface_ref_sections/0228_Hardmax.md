### Hardmax

## 功能

根据 `axis` 找到指定轴，沿指定轴找出第一个最大值，该轴中第一个最大值位置设置为 1，其余位置设置为 0。

## 输入

- **x**：输入 Tensor，输入 Tensor 维度 ≥ `axis`，数据类型：float16、float。

## 属性

- **axis**：数据类型支持 int，默认为 -1，含义：表示 Hardmax 沿哪个维度将执行。

## 输出

- **y**：输出 Tensor，和输入 `x` 同样的数据类型和 shape。

## 约束

使用 atc 工具 `--precision_mode` 参数必须为 `allow_fp32_to_fp16`。

## 支持的 ONNX 版本

Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
