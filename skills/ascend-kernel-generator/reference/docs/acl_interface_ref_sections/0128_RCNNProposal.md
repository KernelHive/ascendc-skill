### RCNNProposal

## 输入

### cls_score_softmax
- **是否必填**：必填
- **数据类型**：float16
- **参数解释**：分类置信度
- **规格限制**：NCHW，shape为[b*topn1,4,1,1]，b表示batch数，topn1表示RPNProposalSSD层的top_n

### bbox_pred
- **是否必填**：必填
- **数据类型**：float16
- **参数解释**：预测框的坐标偏移，用于微调rois
- **规格限制**：NCHW，属性regress_agnostic选项为“true”的时候shape为[b*topn1,8,1,1]（前景和背景），否则为[b*topn1,16,1,1]（背景和3种类别），b表示batch数，topn1表示RPNProposalSSD层的top_n

### rois
- **是否必填**：必填
- **数据类型**：float16
- **参数解释**：预测框的坐标
- **规格限制**：NCHW，shape为[b,topn1,8,1]，b表示batch数，topn1表示RPNProposalSSD层的top_n

### im_info
- **是否必填**：必填
- **数据类型**：float16
- **参数解释**：图像信息
- **规格限制**：NCHW，shape为固定值[b,6,1,1]，b表示batch数

## 属性

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

### num_class
- **是否必填**：非必填
- **数据类型**：int
- **参数解释**：类别数量
- **规格限制**：仅支持3

### rpn_proposal_output_score
- **是否必填**：非必填
- **数据类型**：bool
- **参数解释**：是否输出各类别置信度
- **规格限制**：仅支持true

### regress_agnostic
- **是否必填**：非必填
- **数据类型**：bool
- **参数解释**：是否区分前景类别
- **规格限制**：无

### min_size_h
- **是否必填**：必填
- **数据类型**：float
- **参数解释**：bbox高度最小值，默认值为8.8008
- **规格限制**：无

### min_size_w
- **是否必填**：必填
- **数据类型**：float
- **参数解释**：bbox宽度最小值，默认值为8.8008
- **规格限制**：无

### min_size_mode
- **是否必填**：必填
- **数据类型**：str
- **参数解释**：最小值模式，枚举型，默认值为HEIGHT_OR_WIDTH（HEIGHT_OR_WIDTH=0,HEIGHT_OR_WIDTH=1）
- **规格限制**：无

### threshold_objectness
- **是否必填**：必填
- **数据类型**：float
- **参数解释**：前景阈值
- **规格限制**：取值范围(0,1)

### threshold
- **是否必填**：必填
- **数据类型**：listFloat
- **参数解释**：各类别置信度阈值，size等于num_class
- **规格限制**：取值范围(0,1)

### refine_out_of_map_bbox
- **是否必填**：必填
- **数据类型**：bool
- **参数解释**：是否调整bbox到框内
- **规格限制**：无

### overlap_ratio
- **是否必填**：必填
- **数据类型**：listFloat
- **参数解释**：NMS交并比阈值
- **规格限制**：无

### top_n
- **是否必填**：必填
- **数据类型**：listInt
- **参数解释**：NMS输出框的数目
- **规格限制**：无

### max_candidate_n
- **是否必填**：必填
- **数据类型**：listInt
- **参数解释**：NMS候选框最大数目
- **规格限制**：无

### bsz01
- **是否必填**：非必填
- **数据类型**：float
- **参数解释**：bbox尺寸增加值
- **规格限制**：无

### allow_border
- **是否必填**：非必填
- **数据类型**：float
- **参数解释**：允许超出图像边界像素数目
- **规格限制**：无

### allow_border_ratio
- **是否必填**：非必填
- **数据类型**：float
- **参数解释**：允许超出图像边界面积比例
- **规格限制**：无

## 输出

### bboxes
- **是否必填**：必填
- **数据类型**：float16
- **参数解释**：输出框的信息，包括该框所在图像编号，框的坐标以及各类别置信度：| img_idx | x1 | y1 | x2 | y2 | background_score | cls1_score | cls2_score | cls3_score |
- **规格限制**：NCHW，shape为[b,topn,9,1]，b表示batch数，topn表示本算子的top_n
