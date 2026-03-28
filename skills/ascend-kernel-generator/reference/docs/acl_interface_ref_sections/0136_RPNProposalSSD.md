### RPNProposalSSD

## 输入

### rpn_cls_prob_reshape
- **是否必填**：必填
- **数据类型**：float16
- **参数解释**：前景/背景score信息，feature_map_size=32，C=num_anchors*2
- **规格限制**：NCHW，shape[1,30,32,32]

### rpn_bbox_pred
- **是否必填**：必填
- **数据类型**：float16
- **参数解释**：预测框的坐标，C=num_anchors*4，feature_map_size=32
- **规格限制**：NCHW，shape[1,60,32,32]

### im_info
- **是否必填**：必填
- **数据类型**：float16
- **参数解释**：图像信息
- **规格限制**：NCHW，shape[1,6,1,1]

## 属性

### anchor_height
- **是否必填**：必填
- **数据类型**：listFloat
- **参数解释**：anchor的高
- **规格限制**：无

### anchor_width
- **是否必填**：必填
- **数据类型**：listFloat
- **参数解释**：anchor的宽
- **规格限制**：无

### bbox_mean
- **是否必填**：必填
- **数据类型**：listFloat
- **参数解释**：bbox归一化均值
- **规格限制**：size必须是0或4

### bbox_std
- **是否必填**：必填
- **数据类型**：listFloat
- **参数解释**：bbox归一化标准差
- **规格限制**：size必须是0或4

### intype
- **是否必填**：必填
- **数据类型**：str
- **参数解释**：输入数据类型
- **规格限制**：只支持float16

### top_n
- **是否必填**：必填
- **数据类型**：listInt
- **参数解释**：输出roi的个数
- **规格限制**：不可变，只支持300

### min_size_mode
- **是否必填**：必填
- **数据类型**：str
- **参数解释**：最小值模式
- **规格限制**：无

### min_size_h
- **是否必填**：必填
- **数据类型**：float
- **参数解释**：bbox高度最小值
- **规格限制**：大于0

### min_size_w
- **是否必填**：必填
- **数据类型**：float
- **参数解释**：bbox宽度最小值
- **规格限制**：大于0

### heat_map_a
- **是否必填**：必填
- **数据类型**：listFloat
- **参数解释**：缩放比例
- **规格限制**：大于0

### overlap_ratio
- **是否必填**：必填
- **数据类型**：listFloat
- **参数解释**：nms的iou阈值
- **规格限制**：0到1之间

### threshold_objectness
- **是否必填**：必填
- **数据类型**：float
- **参数解释**：前景阈值
- **规格限制**：0到1之间

### max_candidate_n
- **是否必填**：必填
- **数据类型**：listFloat
- **参数解释**：nms候选框最大数目
- **规格限制**：只支持3000

### refine_out_of_map_bbox
- **是否必填**：必填
- **数据类型**：bool
- **参数解释**：是否调整bbox到框内
- **规格限制**：默认true

### use_soft_nms
- **是否必填**：必填
- **数据类型**：listBool
- **参数解释**：是否使用softnms
- **规格限制**：默认false

### voting
- **是否必填**：必填
- **数据类型**：listBool
- **参数解释**：默认[false]
- **规格限制**：无

### vote_iou
- **是否必填**：必填
- **数据类型**：listFloat
- **参数解释**：默认[0.7]
- **规格限制**：无

## 输出

### rois
- **是否必填**：必填
- **数据类型**：float16
- **参数解释**：输出的感兴趣区域
- **规格限制**：NCHW，shape[n,top_n,8,1]
