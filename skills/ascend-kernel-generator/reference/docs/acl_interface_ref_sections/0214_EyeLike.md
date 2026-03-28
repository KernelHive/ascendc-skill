### EyeLike

## 功能

生成一个2D矩阵，主对角线是1，其他为0。

## 输入

**x**：输入Tensor，shape为二维，用于拷贝Tensor的shape。

**数据类型**：float16、float32、int32、int16、int8、uint8、int64、bool

**数据格式**：支持ND

## 属性

- **dtype**（可选）：指定输出数据类型，支持float16、float、int32、int16、int8、uint8、int64。若未指定，使用首个输入的数据类型。若首个输入的数据类型也未指定，使用默认float。
- **k**（可选）：数据类型支持int，默认是0，表示主对角线被广播成1的索引。如y是输出，则y[i, i+k] = 1。

## 输出

**y**：输出Tensor，和输入x同样的shape。

## 约束

仅支持k=0。

## 支持的 ONNX 版本

Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
