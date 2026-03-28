### DeformableConv2D

## 功能
在给定四维“X”、“filter”和“offsets”Tensor的情况下计算二维可变形卷积。

## 输入
- **X**：输入四维Tensor，格式支持NCHW，数据类型支持float16、float32。
- **filter**：可学习滤波器的四维Tensor，必须与X具有相同的类型，格式支持NCHW，数据类型支持float16、float32。
- **offsets**：x-y坐标偏移量和掩码的四维Tensor，格式支持NCHW，数据类型支持float16、float32。
- **B（可选）**：偏差，输入一维Tensor，shape为[M]，数据类型支持float16、float32，格式支持ND。

## 属性
- **auto_pad（可选）**：支持VALID、NOTSET。
- **dilations**：数据类型为4个整数的列表，指定用于扩张卷积的扩张率，H和W维度取值范围为[1, 255]。
- **groups（可选）**：数据类型为int32，默认值为1，阻止的数量从输入通道到输出通道的阻塞连接数，输入通道和输出通道都必须被groups整除。
- **pads**：数据类型为4个整数的列表，指定顶部、底部、左侧和右侧填充，取值范围为[0, 255]。
- **strides**：数据类型为4个整数的列表，指定沿高度H和宽度W的卷积步长。H和W维度取值范围为[1, 63]，默认情况下，N和C尺寸设置为1。
- **data_format**：数据类型为string，表示输入数据format，默认是NHWC。
- **deformable_groups**：分组卷积通道数，缺省为1。
- **modulated**：数据类型为bool，指定DeformableConv2D版本，true表示v2版本，false表示v1版本，当前只支持true。

## 输出
- **y**：形变卷积输出Tensor，格式为NCHW，数据类型支持float16、float32。

## 约束
- 输入Tensor，W维度取值范围为[1, 4096 / filter_width]，H取值范围为[1, 100000 / filter_height]。
- 权重Tensor，W维度取值范围为[1, 63]，H取值范围为[1, 63]。
- 不支持atc工具`--precision_mode=must_keep_origin_dtype`参数时输入类型：float、float64。

## 支持的 ONNX 版本
Opset v9/v10/v11/v12/v13/v14/v15/v16
