### ROIPooling

## 输入

### x
- **是否必填**：必填
- **数据类型**：float16、float32
- **参数解释**：输入feature Map[batch,C,H,W]
- **规格限制**：h和w随着pooled_h/pooled_w取值不同，其范围也不相同：
  - pooled_h=pooled_w:2,17 → h和w≤50
  - pooled_h=pooled_w:4,5,10-16 → h和w≤70
  - pooled_h=pooled_w:7,8 → h和w≤80
  - pooled_h=pooled_w:3 → h和w≤60
  - pooled_h=pooled_w:18-20 → h和w≤40

### rois
- **是否必填**：必填
- **数据类型**：float16、float32
- **参数解释**：ROIS的输入[batch,5,N]，其中N表示所有Batch中最大roi框的个数向上16对齐后的值。  
  例如：其中某个Batch对应的roi框个数为最大，且值为17，则N=32。  
  N≤6000
- **规格限制**：
  - **默认值**：float16
  - **使用约束**：
    - 每次在填充数据前请先对该内存块执行清零操作
    - roi数据输入给ROIPooling算子前，要先执行clip操作，即需要保证roi的坐标在feature Map的宽高范围内，然后再输入给ROIPooling算子，否则超出feature Map范围的roi框，对应的结果可能会和CPU结果不一致

### roi_actual_num
- **是否必填**：非必填
- **数据类型**：int
- **参数解释**：Tensor的shape为[batch,8]，其中8表示8列数中只有第一列有效（为满足性能加速，进行数据对齐，后面7列数字是补齐的无效数据），表示每个batch实际的rois数量。  
  例如，shape为[4,8]，其中只有首列表示有效数据，存放每个batch的rois数量：
  ```
  [0,0]...[0,7]
  [1,0]...[1,7]
  [2,0]...[2,7]
  [3,0]...[3,7]
  ```
- **规格限制**：
  - **默认值**：int32
  - **使用约束**：如不使用该参数，则推理过程中按rois中的N执行，如使用该参数，则推理过程中按roi_actual_num中的实际roi框个数执行

## 属性

### pooled_h
- **是否必填**：必填
- **数据类型**：int
- **参数解释**：roi_pooling的高，必须大于0
- **规格限制**：[2,20]

### pooled_w
- **是否必填**：必填
- **数据类型**：int
- **参数解释**：roi_pooling的宽，必须大于0
- **规格限制**：[2,20]

### spatial_scale
- **是否必填**：非必填
- **数据类型**：float
- **参数解释**：rois映射到原始Feature Map的缩放比例系数，如果有spatial_scale_h和spatial_scale_w，则以spatial_scale_h和spatial_scale_w为准，如果不提供spatial_scale_h和spatial_scale_w，则在Caffe插件中把spatial_scale转换为spatial_scale_h和spatial_scale_w
- **规格限制**：无

### spatial_scale_h
- **是否必填**：非必填
- **数据类型**：float
- **参数解释**：rois映射到原始Feature Map的高缩放比例系数，默认值为0.0625
- **规格限制**：无

### spatial_scale_w
- **是否必填**：非必填
- **数据类型**：float
- **参数解释**：rois映射到原始Feature Map的宽缩放比例系数，默认值为0.0625
- **规格限制**：无

## 输出

### y
- **是否必填**：必填
- **数据类型**：float16、float32
- **参数解释**：从feature map中根据rois的坐标crop出对应部分后，再根据输出size的配置进行max pooling[batch*N,C,pooled_h,pooled_w]
- **规格限制**：无
