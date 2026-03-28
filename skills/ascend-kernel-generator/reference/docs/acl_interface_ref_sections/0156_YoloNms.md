### YoloNms

## 输入

### x1

- **是否必填**：必填
- **数据类型**：float16
- **参数解释**：Yolov5FourInputDecodeBox的输出结果
- **规格限制**：NCHW

## 属性

### shape

- **是否必填**：必填
- **数据类型**：int
- **参数解释**：输入Yolov5FourInputDecodeBox算子的4个feature_map的h*w

### thresh

- **是否必填**：必填
- **数据类型**：float16
- **参数解释**：NMS的阈值
- **规格限制**：0.0-1.0之间

### num_anchor

- **是否必填**：必填
- **数据类型**：int
- **参数解释**：anchor数
- **规格限制**：大于0

### num_class

- **是否必填**：必填
- **数据类型**：int
- **参数解释**：类别数
- **规格限制**：1到80类

### total_output_proposal_num

- **是否必填**：必填
- **数据类型**：int
- **参数解释**：每个检测的类别要输出的框的个数
- **规格限制**：5到50之间

## 输出

### y

- **是否必填**：必填
- **数据类型**：float16
- **参数解释**：ssmh网络后处理进行nms之后的输出结果，[total_output_proposal_num*class_num, 8]
- **规格限制**：NCHW，输出的框是按类别排布的，[total_output_proposal_num*class_num, :4]是坐标，[total_output_proposal_num*class_num, 4]是得分，[total_output_proposal_num*class_num, 5:]是保留位
