### AscendAntiQuant

## 功能
对输入x进行反重量化操作。

## 输入
- **x**：输入Tensor，数据类型支持int8，数据格式支持ND。

## 属性
- **offset**：float数据类型，指定偏移量。
- **scale**：float数据类型，指定缩放比例。
- **sqrt_mode**：bool数据类型，指定是否在scale上进行平方根计算，默认false（可选）。

## 输出
- **y**：输出Tensor，数据类型支持float16，数据格式支持ND。

## 约束与限制
- Atlas 推理系列产品的入参scale、入参offset和出参y，数据格式不支持bfloat16。
- Atlas 推理系列产品的入参x，数据格式不支持int4、int32。

## 支持的 ONNX 版本
Opset v8/v9/v10/v11/v12/v13/v14/v15/v16
