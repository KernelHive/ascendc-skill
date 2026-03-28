### YoloV2DetectionOutput

## 输入

### coord_data
- **是否必填**: 必填
- **数据类型**: float16、float32
- **参数解释**: [batch,coords *boxes,height,width]，表示预测的coords
- **规格限制**: float16，height*width*Dtype_Size>=32 Byte

### obj_prob
- **是否必填**: 必填
- **数据类型**: float16、float32
- **参数解释**: [batch,boxes,height,width]，此处每个anchor的obj的数值只有1个
- **规格限制**: float16

### classes_prob
- **是否必填**: 必填
- **数据类型**: float16、float32
- **参数解释**: [batch,boxes*classes,height,width]，为了方便AICORE计算已将每个anchor的score向16取整
- **规格限制**: float16，height*width*Dtype_Size>=32 Byte

### img_info
- **是否必填**: 必填
- **数据类型**: float16、float32
- **参数解释**: 原图信息，[batch,4]，4表示netH、netW、scaleH、scaleW四个维度；其中netH、netW为网络模型输入的HW，scaleH、scaleW为原始图片的HW
- **规格限制**: float16

## 属性

### biases
- **是否必填**: 必填
- **数据类型**: ListFloat
- **参数解释**: [boxes,2]，其中2分别表示x(w)和y(h)方向
- **规格限制**: 无

### boxes
- **是否必填**: 非必填
- **数据类型**: int
- **参数解释**: 每个grid的anbox的数量，默认值5
- **规格限制**: 无

### coords
- **是否必填**: 非必填
- **数据类型**: int
- **参数解释**: coords的数量，固定为4，表示x、y、h、w
- **规格限制**: 固定为4

### classes
- **是否必填**: 非必填
- **数据类型**: int
- **参数解释**: 类别数，默认为20
- **规格限制**: 最大1024

### relative
- **是否必填**: 非必填
- **数据类型**: bool
- **参数解释**: 在correct_region_boxes中表示是否为相对值，True
- **规格限制**: True或者False

### obj_threshold
- **是否必填**: 非必填
- **数据类型**: float
- **参数解释**: 有物体概率的阈值，对应于clsProb中的阈值，默认值为0.5
- **规格限制**: [0,1]

### pre_nms_topn
- **是否必填**: 非必填
- **数据类型**: int
- **参数解释**: pre_nms_topn对应于multiClassNMS（对于每个类别，取前pre_nms_topn个数量进行处理，SoC最大支持512；Mini/Cloud最大支持1024，默认值512）
- **规格限制**: 最大1024

### post_nms_topn
- **是否必填**: 非必填
- **数据类型**: int
- **参数解释**: 经过nms之后返回全部还是postTopK个框，最大为1024；默认值512
- **规格限制**: 最大1024

### score_threshold
- **是否必填**: 非必填
- **数据类型**: float
- **参数解释**: 每个类别的阈值，默认值为0.5
- **规格限制**: [0,1]

### iou_threshold
- **是否必填**: 非必填
- **数据类型**: float
- **参数解释**: 交并比（Intersection over Union）阈值，默认值是0.45
- **规格限制**: [0,1]

## 输出

### box_out
- **是否必填**: 必填
- **数据类型**: float16、float32
- **参数解释**: [batch,6*post_nms_topn]，其中的6表示x1、y1、x2、y2、score、label（类别），实际按box_out_num的数量输出
- **规格限制**: float16

### box_out_num
- **是否必填**: 必填
- **数据类型**: int32
- **参数解释**: [batch,8,1,1]，8表示8列数中只有第一列有效（为满足性能加速，进行数据对齐，后面7列数字是补齐的无效数据），表示每个batch中有效框的数量，每个batch中有效框的数量最大为1024

例如，Shape为[4,8]，其中只有首列表示有效数据，存放每个batch的有效框数量：

```
[0,0]...[0,7]
[1,0]...[1,7]
[2,0]...[2,7]
[3,0]...[3,7]
```

- **规格限制**: 无
