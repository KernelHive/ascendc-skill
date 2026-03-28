### Reshape

## 输入

### x
- **是否必填**：必填
- **数据类型**：float16、float32、int8、uint8、int16、uint16、int32、uint32、int64、uint64
- **参数解释**：输入Tensor
- **规格限制**：无

### shape
- **是否必填**：非必填
- **数据类型**：int32、int64
- **参数解释**：指示输出的维度大小，0表示跟bottom一致，-1表示该维度由输入的blob及输出的其他维度决定
- **规格限制**：无

## 属性

### axis
- **是否必填**：必填
- **数据类型**：int
- **参数解释**：默认为0，表示shape中第一个数值与输出的第几个起始维度对应
- **规格限制**：[-rank(x), rank(x))

### num_axes
- **是否必填**：非必填
- **数据类型**：int
- **参数解释**：计算输出Shape的维度数：
  - 当num_axes == -1，输出Tensor的shape.size() = shape.size() + axis
  - 当num_axes <> -1时，输出Tensor的shape.size() = shape.size() + tensor.size() - num_axes
- **规格限制**：支持[-1, rank(x))范围内可配，默认值-1表示对axis起始的所有轴进行变换

## 输出

### y
- **是否必填**：必填
- **数据类型**：float16、float32、int8、uint8、int16、uint16、int32、uint32、int64、uint64
- **参数解释**：维度变换后的输出Tensor
- **规格限制**：无
