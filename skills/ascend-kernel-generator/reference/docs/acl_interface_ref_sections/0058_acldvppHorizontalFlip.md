## acldvppHorizontalFlip

## 支持的产品型号

- Atlas A2 训练系列产品/Atlas A2 推理系列产品
- Atlas A3 训练系列产品/Atlas A3 推理系列产品

## 功能说明

算子功能：对图像做水平翻转。

## 函数原型

每个算子有两段接口，必须先调用 `acldvppHorizontalFlipGetWorkspaceSize` 接口获取入参并根据计算流程计算所需 workspace 大小，再调用 `acldvppHorizontalFlip` 接口执行计算。两段式接口如下：

- **第一段接口：**
  ```c
  acldvppStatus acldvppHorizontalFlipGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
  ```

- **第二段接口：**
  ```c
  acldvppStatus acldvppHorizontalFlip(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
  ```

## acldvppHorizontalFlipGetWorkspaceSize

### 参数说明

- **self**：表示算子输入 Tensor，需调用 `aclCreateTensor` 接口创建 `aclTensor` 类型的数据（数据存放在 Device 侧），输入 Tensor 的 dataType 支持 `UINT8`/`FLOAT`、Format 支持 `NCHW`/`NHWC`、不支持非连续的 Tensor，同时 N 支持 1 或空、C 支持 1 或 3（1 表示输入 GRAY 图，3 表示输入 RGB 图）。
- **out**：表示算子输出 Tensor，需调用 `aclCreateTensor` 接口创建 `aclTensor` 类型的数据（数据存放在 Device 侧），输出 Tensor 的 dataType 支持 `UINT8`/`FLOAT`、Format 支持 `NCHW`/`NHWC`、不支持非连续的 Tensor，同时 N 只支持 1、C 支持 1 或 3（1 表示 GRAY 图，3 表示 RGB 图），dataType、Format、Shape 需要和 self 一致。
- **workspaceSize**：返回用户需要在 Device 侧申请的 workspace 大小。
- **executor**：返回 op 执行器，包含了算子计算流程。

### 返回值

返回 `acldvppStatus` 状态码，具体请参见 6.2 acldvpp 返回码。

## acldvppHorizontalFlip

### 参数说明

- **workspace**：需调用 `aclrtMalloc` 接口申请 Device 内存，内存大小为 `workspaceSize`，`aclrtMalloc` 接口输出的内存地址在此处传入。
- **workspaceSize**：与 `acldvppHorizontalFlipGetWorkspaceSize` 接口获取的 `workspaceSize` 保持一致。
- **executor**：op 执行器，包含了算子计算流程，与 `acldvppHorizontalFlipGetWorkspaceSize` 接口的 `executor` 保持一致。
- **stream**：指定执行任务的 Stream，可复用已创建的 Stream 节省资源或调用 `aclrtCreateStream` 接口创建 Stream，再作为入参在此处传入。

### 返回值

返回 `acldvppStatus` 状态码，具体请参见 6.2 acldvpp 返回码。

## 约束说明

支持图像分辨率范围在 [6×4 ~ 4096×8192]。
