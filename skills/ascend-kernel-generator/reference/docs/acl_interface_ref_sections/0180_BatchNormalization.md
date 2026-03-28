### BatchNormalization

## 功能
对输入 Tensor 进行批量归一化操作。

## 输入
- **X**：输入 Tensor，数据类型支持 float16、float 的 4D tensor，数据格式支持 ND。
- **scale**（可选）：输入 Tensor，数据类型支持 float，指定尺度因子，数据格式支持 ND。
- **B**（可选）：输入 Tensor，数据类型支持 float，指定偏移量，数据格式支持 ND。
- **mean**（可选）：输入 Tensor，数据类型支持 float，指定均值，数据格式支持 ND。
- **var**（可选）：输入 Tensor，数据类型支持 float，指定方差，数据格式支持 ND。

## 属性
- **epsilon**（可选）：数据类型为 float，指定一个小值与 var 相加，以避免除以 0，默认为 0.0001。
- **momentum**：数据类型为 float，该参数暂不支持。

## 输出
- **Y**：标准化之后的 Tensor，数据类型为 float16 或 float，数据格式支持 ND。
- **running_mean**（可选）：数据类型为 float 的 Tensor，均值，数据格式支持 ND。
- **running_var**（可选）：数据类型为 float 的 Tensor，方差，数据格式支持 ND。
- **saved_mean**（可选）：数据类型为 float 的 Tensor，在训练过程中使用已保存的平均值来加快梯度计算（v14 后官方不支持），数据格式支持 ND。
- **saved_var**（可选）：数据类型为 float 的 Tensor，在训练过程中使用已保存的方差来加快梯度计算（v14 后官方不支持），数据格式支持 ND。

## 约束与限制
无。

## 支持的 ONNX 版本
Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
