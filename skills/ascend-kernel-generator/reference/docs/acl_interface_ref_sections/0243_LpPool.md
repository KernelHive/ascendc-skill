### LpPool

## 功能
Lp范数池化。接收输入Tensor x，并根据内核大小、步幅大小和填充长度对Tensor进行Lp池化。Lp池化包括计算输入Tensor的一个子集的所有值的Lp范数，并根据内核大小对数据进行降采样，将其放入输出Tensor y以供进一步处理。

## 输入
- **x**：输入Tensor，数据类型支持float16。

## 属性
- **auto_pad**：数据类型为string，默认为`NOTSET`，支持：`NOTSET`、`SAME_UPPER` 或者 `VALID`。
- **kernel_shape**：数据类型为int列表，kernel每个轴上的尺寸。
- **p**：数据类型为int，范数，用于对输入数据进行池化的Lp范数的p值，默认为2。
- **pads**：数据类型为int列表。沿每个空间轴的开始和结束的填充，它可以取大于或等于0的任何值。该值表示在相应轴的开始和结束部分添加的像素数。pads格式应如下所示：`[x1_begin, x2_begin…x1_end, x2_end,…]`，其中xi_begin是在轴i的开始处添加的像素数，xi_end是在轴i的结束处添加的像素数。此属性不能与auto_pad属性同时使用。如果不出现，每个空间轴的开始和结束的填充默认为0。
- **strides**：数据类型为int列表。沿每个空间轴的步幅。如果不出现，每个空间轴的步幅默认为1。

## 说明
auto_pad属性值支持`SAME_UPPER`而不支持`SAME_LOWER`，原因是`SAME_LOWER`和`SAME_UPPER`统一使用的TBE的SAME属性，即TBE算子没有根据这个属性区分pad的填充位置，可能会带来精度问题。

## 输出
- **y**：输出Tensor，数据类型和shape与输入一致。

## 约束与限制
无。

## 支持的 ONNX 版本
Opset v11/v12/v13/v14/v15/v16/v17/v18
