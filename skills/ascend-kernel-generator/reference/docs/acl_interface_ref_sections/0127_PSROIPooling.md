### PSROIPooling

## 输入

### x
- **是否必填**: 必填
- **数据类型**: float16
- **参数解释**: feature map, [batch, C, H, W]，其中的 C = rois_num * group_size²
- **规格限制**: 无

### rois
- **是否必填**: 必填
- **数据类型**: float16
- **参数解释**: Shape 是 [batch, 5, rois_num]，其中的 5 表示 (batchID, x1, y1, x2, y2)
- **规格限制**: 无

## 属性

### spatial_scale
- **是否必填**: 必填
- **数据类型**: float
- **参数解释**: 进行 ROIPooling 特征输入的尺寸相比原始输入的比例
- **规格限制**: 大于 0

### output_dim
- **是否必填**: 必填
- **数据类型**: int
- **参数解释**: 输出 channels，必须大于 0
- **规格限制**: 支持输出 ROI 的 channel 维度大小

### group_size
- **是否必填**: 必填
- **数据类型**: int
- **参数解释**: 输出 feature 的高和宽
- **规格限制**: 支持输出 ROI 的 H 和 W 大小，H = W；要求原始的 input_channel = out_channel * group_size²

## 输出

### y
- **是否必填**: 必填
- **数据类型**: float16
- **参数解释**: 类似 ROIPooling，不同位置的输出来自不同 channel 的输入 [batch * rois_num, output_dim, group_size, group_size]
- **规格限制**: 无
