### DepthToSpace

## 功能

将数据从深度重新排列（置换）到空间数据块中，是 SpaceToDepth 的反向转换。

## 输入

**input**：输入 Tensor，shape 为 `[N, C, H, W]`，其中：

- N 是批处理轴
- C 是通道或深度
- H 是高度
- W 是宽度

支持的数据类型：float16、float32、double、int32、uint8、int16、int8、int64、uint16、uint32、uint64、bfloat16。

## 属性

- **blocksize**：数据类型为 int，大于等于 2，指定被移动的块的大小。
- **mode**：数据类型为 string，指定是 depth-column-row 还是 column-row-depth 排列，默认为 DCR。
- **data_format**（可选）：用于指定数据格式。默认为 “NHWC”。

## 输出

**output**：输出 Tensor，shape 为 `[N, C/(blocksize * blocksize), H * blocksize, W * blocksize]`，数据类型与输入一致。

## 约束与限制

无。

## 支持的 ONNX 版本

Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
