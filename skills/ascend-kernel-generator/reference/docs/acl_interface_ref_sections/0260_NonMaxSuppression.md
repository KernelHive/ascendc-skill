### NonMaxSuppression

## 功能

从众多重叠的候选边界框中筛选出最优的边界框，过滤掉与先前选定的框有较高重叠的“交集-并集”(IOU)框。移除得分小于 `score_threshold` 的边界框。边界框格式由属性 `center_point_box` 表示。

该算法与坐标系原点无关，对坐标系的正交变换和平移操作保持不变。输出 `selected_indices` 为整数集，可通过 Gather 或 GatherND 操作获取对应边框坐标。

## 输入

- **boxes**：输入 Tensor，数据类型：float，存储候选边界框信息。
- **scores**：输入 Tensor，数据类型：float，候选边界框对应的置信度分数。
- **max_output_boxes_per_class**（可选）：输入常量值，数据类型：int64，每种类最多可选取的边界框数量。
- **iou_threshold**（可选）：输入常量值，数据类型：float，IOU 重叠判断阈值。
- **score_threshold**（可选）：输入常量值，数据类型：float，置信度阈值。

## 属性

**center_point_box**：数据类型为 int，默认值为 0，决定了边界框格式。

- 等于 0 时，主要用于 TF 模型，数据以 `(y1, x1, y2, x2)` 形式提供，其中 `(y1, x1)`、`(y2, x2)` 是对角线框角坐标，需要用户自行保证 `x1 < x2`、`y1 < y2`。
- 等于 1 时，主要用于 PyTorch 模型，框数据以 `(x_center, y_center, width, height)` 形式提供。

## 输出

**selected_indices**：输出 Tensor，数据类型：int64。

## 约束与限制

- Atlas 200/300/500 推理产品和 Atlas 训练系列产品对于输入 `boxes` 和 `scores` 的数据类型仅支持 float16。
- `max_output_boxes_per_class` 数值大于 700 时，可能会引发硬件资源不足问题。

## 支持的 ONNX 版本

Opset v11/v12/v13/v14/v15/v16/v17/v18。
