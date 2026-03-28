### BatchMultiClassNMS

## 功能
计算输入 Tensor `boxes` 和 `scores` 的 NMS，支持多个 batch 和 class，抑制非极大值元素，保留极大值元素。

## 输入
- **boxes**：输入 Tensor，shape 为 `[batch, num_anchors, num_classes, 4]`，其中：
  - `batch`：图片的 batch 大小
  - `num_anchors`：框数
  - `num_classes`：检测类别
  - `4` 表示坐标顺序为 `x0`、`x1`、`y0`、`y1`
  - 数据类型：float16、float32

- **scores**：输入 Tensor，shape 为 `[batch, num_anchors, num_classes]`
  - 数据类型：float16、float32

- **clip_window**（可选）：输入 Tensor，表示窗口大小，shape 为 `[batch, 4]`
  - 数据类型：float16、float32

- **num_valid_boxes**（可选）：输入 Tensor，表示每个批次的有效框编号
  - 数据类型：int32

## 属性
- **score_threshold**：float 类型，指定分数筛选器
- **iou_threshold**：float 类型，表示 IOU 重叠判断阈值
- **max_size_per_class**：int 类型，指定每个类的 NMS 输出编号
- **max_total_size**：int 类型，指定每批的 NMS 输出数
- **change_coordinate_frame**（可选）：bool 类型，表示是否在裁剪后标准化坐标
- **transpose_box**（可选）：bool 类型，表示是否在此操作之前插入转置（必须为 `false`）

## 输出
- **nmsed_boxes**：输出 Tensor，指定每批的输出 NMS 框
  - 数据类型：float16、float32

- **nmsed_scores**：输出 Tensor，指定每批的输出 NMS 分数
  - 数据类型：float16、float32

- **nmsed_classes**：输出 Tensor，指定每个批次的输出 NMS 类
  - 数据类型：float16、float32

- **nmsed_num**：输出 Tensor，指定 `nmsed_boxes` 的有效数量
  - 数据类型：int32

## 约束与限制
无

## 支持的 ONNX 版本
Opset v8/v9/v10/v11/v12/v13/v14/v15/v16
