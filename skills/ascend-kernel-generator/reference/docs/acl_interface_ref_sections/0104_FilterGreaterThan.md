### FilterGreaterThan

## 输入

### argmax_score

- **是否必填**：必填
- **数据类型**：float16
- **参数解释**：argmax后的置信度
- **规格限制**：NCHW

## 属性

### scorethresh

- **是否必填**：非必填
- **数据类型**：float32
- **参数解释**：置信度阈值
- **规格限制**：defaultValue=1

## 输出

### output_filter

- **是否必填**：必填
- **数据类型**：float16
- **参数解释**：过滤后的置信度
- **规格限制**：NCHW

### output_cnt

- **是否必填**：必填
- **数据类型**：int32
- **参数解释**：在阈值范围内的个数
- **规格限制**：NCHW
