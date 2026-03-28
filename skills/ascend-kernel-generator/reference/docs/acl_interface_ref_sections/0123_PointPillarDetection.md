### PointPillarDetection

## 输入

### x1
- **是否必填**：必填
- **数据类型**：float16
- **参数解释**：PointPillarDecodeBox的输出框结果
- **规格限制**：NCHW

### x2
- **是否必填**：必填
- **数据类型**：float16
- **参数解释**：ClassCal的输出class_out
- **规格限制**：NCHW

### x3
- **是否必填**：必填
- **数据类型**：float16
- **参数解释**：ClassCal的输出heading_out
- **规格限制**：NCHW

## 属性

### thresh
- **是否必填**：必填
- **数据类型**：float16
- **参数解释**：nms的阈值
- **规格限制**：0.0-1.0之间

### per_class_num
- **是否必填**：可选
- **数据类型**：int64
- **参数解释**：每个类别保留的候选框的个数
- **规格限制**：20 <= per_class_num <= 60

## 输出

### y
- **是否必填**：必填
- **数据类型**：float16
- **参数解释**：PointPillar的解析框的输出
- **规格限制**：NCHW
