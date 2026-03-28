### SSDDetectionOutput

## 输入

### bbox_delta
- **是否必填**: 必填
- **数据类型**: float16、float32
- **参数解释**: 框位置偏移数据[batch,N*num_loc_classes*4]
  其中 num_loc_classes = share_location==True ? 1 : num_classes，4表示（xmin,ymin,xmax,ymax）
- **规格限制**: float16

### score
- **是否必填**: 必填
- **数据类型**: float16、float32
- **参数解释**: 置信度数据[batch,N*num_classes]
- **规格限制**: float16

### anchors
- **是否必填**: 必填
- **数据类型**: float16、float32
- **参数解释**: 预选框数据[batch,2,N*4]或者[1,2,N*4]，其中的2分别表示box和variance，如果variance_encoded_in_target==True，可以不需要variance（即里面的2可以更改为1），4表示（xmin,ymin,xmax,ymax）
- **规格限制**: float16

## 属性

### num_classes
- **是否必填**: 必填
- **数据类型**: int
- **参数解释**: 要预测的类数，当background_label_id==-1时，必须大于等于1，如果background_label_id>=0时，必须大于等于2；最大支持1024，默认值为2
- **规格限制**: 最大支持1024

### share_location
- **是否必填**: 非必填
- **数据类型**: bool
- **参数解释**: 表示不同类间共享框位置，默认值True
- **规格限制**: 无

### background_label_id
- **是否必填**: 非必填
- **数据类型**: int
- **参数解释**: Background label id，必须为大于等于-1，默认为0
- **规格限制**: 大于等于-1

### iou_threshold
- **是否必填**: 非必填
- **数据类型**: float
- **参数解释**: 交并比（Intersection over Union）阈值，(0,1]，默认值为0.3
- **规格限制**: (0,1]

### top_k
- **是否必填**: 非必填
- **数据类型**: int
- **参数解释**: 在nms步骤前每个图像要保留的总bbox数，(0,1024]，默认为200
- **规格限制**: (0,1024]

### eta
- **是否必填**: 非必填
- **数据类型**: float
- **参数解释**: nms参数，默认值为1（只支持1）
- **规格限制**: 只支持1

### variance_encoded_in_target
- **是否必填**: 非必填
- **数据类型**: bool
- **参数解释**: 方差是否被编码，默认为False
- **规格限制**: 无

### code_type
- **是否必填**: 非必填
- **数据类型**: int
- **参数解释**: bbox的编解码方式，默认值为1，corner=1 center_size=2 corner_size=3
- **规格限制**: 支持1,2,3

### keep_top_k
- **是否必填**: 非必填
- **数据类型**: int
- **参数解释**: 在nms步骤后每个图像要保留的总bbox数，(0,1024]，默认为200，如果为-1表示NMS后的框都保留
- **规格限制**: (0,1024]

### confidence_threshold
- **是否必填**: 非必填
- **数据类型**: float
- **参数解释**: 仅考虑置信度大于阈值的检测 [0,1]
- **规格限制**: [0,1]

## 输出

### out_boxnum
- **是否必填**: 必填
- **数据类型**: int32
- **参数解释**: 输出框的个数[batch,8]，其中8表示8列数中只有第一列有效（为满足性能加速，进行数据对齐，后面7列数字是补齐的无效数据），表示每个batch的第一列是实际框的个数
  例如，Shape为[4,8]，其中只有首列表示有效数据，存放每个batch的实际框数量：
  ```
  [0,0]...[0,7]
  [1,0]...[1,7]
  [2,0]...[2,7]
  [3,0]...[3,7]
  ```
- **规格限制**: 无

### y
- **是否必填**: 必填
- **数据类型**: float16、float32
- **参数解释**: 输出框数据，[batch,len,8]，其中8表示(batchID,label(classID),score(类别概率)，xmin,ymin,xmax,ymax,null)；len是keep_top_k 128对齐后的取值（如batch为2，keep_top_k为200，则最后输出shape为(2,256,8)），前256*8个数据为第一个batch的结果
- **规格限制**: float16
