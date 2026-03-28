### MaxPool

## 功能

MaxPool接收输入Tensor x，并根据内核大小、步幅大小和填充长度对Tensor进行最大池化。最大池化包括根据内核大小计算输入Tensor的一个子集的所有值的最大值，并将数据降采样到输出Tensor y以供进一步处理。输出的空间形状根据是否使用显式填充、是否使用pads或是否使用auto_pad（自动填充）来计算。

## 输入

**x**：输入Tensor，数据类型支持float、float16，格式为NCHW。

## 属性

**auto_pad**（可选）：支持SAME_UPPER、SAME_LOWER、VALID、NOTSET。

**storage_order**：暂不支持该参数。

**kernel_shape**（必选）：
- kernel_shape[0]：数据类型为int32，指定沿H维度的窗口大小，取值范围为[1, 32768]，默认为1
- kernel_shape[1]：数据类型为int32，指定沿W维度的窗口大小，取值范围为[1, 32768]，默认为1

**strides**（可选）：
- strides[0]：数据类型为int32，指定沿H维度的步长，默认为1
- strides[1]：数据类型为int32，指定沿W维度的步长，默认为1

**pads**（可选）：
- pads[0]：数据类型为int32，指定顶部padding，默认为0
- pads[1]：数据类型为int32，指定底部padding，默认为0
- pads[2]：数据类型为int32，指定左部padding，默认为0
- pads[3]：数据类型为int32，指定右部padding，默认为0

**ceil_mode**（可选）：数据类型为int32，取值：0（floor模式），1（ceil模式），默认为0。

## 输出

**y**：输出Tensor，数据类型和输入一致。格式为NCHW。

## 约束与限制

- strides[0]或者strides[1]取值步长大于63时，会使用AI CPU计算，性能会下降
- kernel_shape[0]或kernel_shape[1]取值超过[1,255]，或者kernel_shape[0] × kernel_shape[1] > 256时，会使用AI CPU计算，导致性能下降
- 1 <= input_w <= 4096
- 当输入Tensor的N是一个质数时，N应小于65535
- 2D Tensor输入不支持dilations
- auto_pad属性是VALID时，ceil_mode属性值必须为0
- pads属性和auto_pad属性不可同时使用

## 支持的 ONNX 版本

Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
