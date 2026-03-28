### ClassCal

## 输入

### class
- **是否必填**：必填
- **数据类型**：float16
- **参数解释**：类别信息
- **规格限制**：NCHW，[N, Anchors * class_num, H, W]

### heading
- **是否必填**：必填
- **数据类型**：float16
- **参数解释**：head信息
- **规格限制**：NCHW，[N, Anchors * 2, H, W]

## 属性

### use_sigmoid
- **是否必填**：必填
- **数据类型**：bool
- **参数解释**：默认为true，使用sigmoid，false则使用softmax

## 输出

### class_out
- **是否必填**：必填
- **数据类型**：float16
- **参数解释**：类别信息处理的输出
- **规格限制**：[N, Anchors, class_num, Aligned16(H*W)]

### heading_out
- **是否必填**：必填
- **数据类型**：float16
- **参数解释**：head处理信息的输出
- **规格限制**：[N, Anchors, 1, Aligned16(H*W)]
