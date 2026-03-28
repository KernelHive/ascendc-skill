### RandomUniform

## 功能
生成具有从均匀分布绘制的随机值的 Tensor。

## 属性
- **dtype**：数据类型为 int，指明输出类型。
- **high**：数据类型为 float，指明上边界。
- **low**：数据类型为 float，指明下边界。
- **seed**（可选）：数据类型为 int，随机种子。
- **shape**：数据类型为 int，输出的形状。

## 输出
**y**：输出 Tensor，数据类型与属性 `dtype` 指定类型一致。

## 约束与限制
无。

## 支持的 ONNX 版本
Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18。
