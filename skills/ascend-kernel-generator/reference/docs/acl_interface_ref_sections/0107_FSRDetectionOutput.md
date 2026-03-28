### FSRDetectionOutput

## 输入

### rois
- **是否必填**: 必填
- **数据类型**: float16、float32
- **参数解释**: Proposal的输出值，`[batch, 5, max_rois_num]`，其中的5表示 `(batchID, x1, y1, x2, y2)`，`max_rois_num` 是每个batch最大的rois的个数，必须是16的倍数，实际每个batch按 `actual_rois_num` 处理数量
- **规格限制**: float16

### bbox_delta
- **是否必填**: 必填
- **数据类型**: float16、float32
- **参数解释**: bbox_delta的Tensor，`[total_rois, num_classes * 4]`，这里的4表示 `(delta_x, delta_y, delta_w, delta_h)`，`total_rois` 指实际rois的总数
- **规格限制**: float16

### score
- **是否必填**: 必填
- **数据类型**: float16、float32
- **参数解释**: `[total_rois, num_classes]`，每个类别的概率（其中classes为0表示背景概率），`total_rois` 指实际rois的总数
- **规格限制**: float16

### im_info
- **是否必填**: 非必填
- **数据类型**: float16、float32
- **参数解释**: 输入图像的高和宽
- **规格限制**: 无

### actual_rois_num
- **是否必填**: 非必填
- **数据类型**: int
- **参数解释**: `[batch_rois, 8]`，其中8表示8列数中只有第一列有效（为满足性能加速，进行数据对齐，后面7列数字是补齐的无效数据），表示每个batch实际输出的rois数量。例如，Shape为 `[4, 8]`，其中只有首列表示有效数据，存放每个batch的rois数量：
  ```
  [0,0]...[0,7]
  [1,0]...[1,7]
  [2,0]...[2,7]
  [3,0]...[3,7]
  ```
- **规格限制**: 无

## 属性

### batch_rois
- **是否必填**: 非必填
- **数据类型**: int
- **参数解释**: rois中含有的batch数，默认值为1（接口保留，实际从shape中取值）
- **规格限制**: 无

### num_classes
- **是否必填**: 必填
- **数据类型**: int
- **参数解释**: 类的数量，包括背景
- **规格限制**: 无

### score_threshold
- **是否必填**: 必填
- **数据类型**: float
- **参数解释**: score的阈值
- **规格限制**: 无

### iou_threshold
- **是否必填**: 必填
- **数据类型**: float
- **参数解释**: 交并比（Intersection over Union）阈值
- **规格限制**: 无

## 输出

### actual_bbox_num
- **是否必填**: 必填
- **数据类型**: int32
- **参数解释**: `[batch, num_classes]`，数据类型是int32，实际输出的bbox数量
- **规格限制**: 无

### box
- **是否必填**: 必填
- **数据类型**: float16、float32
- **参数解释**: 实际输出的Proposal，`[batch, numBoxes, 8]`，其中的8表示 `[x1, y1, x2, y2, score, label, batchID, NULL]`，`numBoxes` 的最大值是1024，即取 `min(输入框最大数量, 1024)`
- **规格限制**: float16
