### ScatterNdV

## 输入

### indices

- **是否必填**：必填
- **数据类型**：int32、int64
- **参数解释**：索引
- **规格限制**：NCHW，indices_c == 3（坐标的维度为3维）

### updates

- **是否必填**：必填
- **数据类型**：float16、float32
- **参数解释**：用于索引的张量
- **规格限制**：
  - NCHW
  - update_h * update_w - 1 <= 65535
  - update_5c <= 61184（数据的通道向上取整到16倍数要小于61184）
  - update_c == dst_c（更新的数据的通道要与最终输出的通道相等）
  - update_n == indices_n
  - update_h * update_w == indice_h * indice_w

## 属性

### reshape_n

- **是否必填**：必填
- **数据类型**：int
- **参数解释**：n通道值

### reshape_c

- **是否必填**：必填
- **数据类型**：int
- **参数解释**：c通道值
- **规格限制**：reshape_c == dst_c

### reshape_h

- **是否必填**：必填
- **数据类型**：int
- **参数解释**：h通道值
- **规格限制**：reshape_h == dst_h

### reshape_w

- **是否必填**：必填
- **数据类型**：int
- **参数解释**：w通道值
- **规格限制**：reshape_w == dst_w

## 输出

### output

- **是否必填**：必填
- **数据类型**：float16、float32
- **参数解释**：索引后输出张量
- **规格限制**：NCHW，dst_h <= 2048 and dst_w <= 2048
