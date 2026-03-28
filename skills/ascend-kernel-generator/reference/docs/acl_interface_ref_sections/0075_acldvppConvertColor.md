## acldvppConvertColor

## 支持的产品型号

- Atlas A2 训练系列产品 / Atlas A2 推理系列产品
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品

## 功能说明

算子功能：更改图像的色彩空间。

## 函数原型

每个算子有两段接口，必须先调用 `acldvppConvertColorGetWorkspaceSize` 接口获取入参并根据计算流程计算所需 workspace 大小，再调用 `acldvppConvertColor` 接口执行计算。

两段式接口如下：

- **第一段接口：**
  ```c
  acldvppStatus acldvppConvertColorGetWorkspaceSize(const aclTensor *self, acldvppConvertMode convertMode, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
  ```

- **第二段接口：**
  ```c
  acldvppStatus acldvppConvertColor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
  ```

## acldvppConvertColorGetWorkspaceSize

### 参数说明

- **self**：表示算子输入 Tensor，需调用 `aclCreateTensor` 接口创建 `aclTensor` 类型的数据（数据存放在 Device 侧），输入 Tensor 的 dataType 支持 `UINT8`/`FLOAT`、Format 支持 `NHWC`、不支持非连续的 Tensor，同时 N 支持 1 或空、C 支持 1 或 3 或 4（1 表示输入 GRAY 图，3 表示输入 RGB 图，4 表示包含 alpha 通道）。
- **convertMode**：图像色彩空间转换的模式（枚举值与 OpenCV 保持一致）。若设置的转换模式涉及 alpha 通道，则只支持 `UINT8` 类型输入和输出数据。convertMode 参数的取值范围如下：

  - `COLOR_BGR2BGRA`：将 BGR 图像转换为 BGRA 图像。
  - `COLOR_RGB2RGBA`：将 RGB 图像转换为 RGBA 图像。
  - `COLOR_BGRA2BGR`：将 BGRA 图像转换为 BGR 图像。
  - `COLOR_RGBA2RGB`：将 RGBA 图像转换为 RGB 图像。
  - `COLOR_BGR2RGBA`：将 BGR 图像转换为 RGBA 图像。
  - `COLOR_RGB2BGRA`：将 RGB 图像转换为 BGRA 图像。
  - `COLOR_RGBA2BGR`：将 RGBA 图像转换为 BGR 图像。
  - `COLOR_BGRA2RGB`：将 BGRA 图像转换为 RGB 图像。
  - `COLOR_BGR2RGB`：将 BGR 图像转换为 RGB 图像。
  - `COLOR_RGB2BGR`：将 RGB 图像转换为 BGR 图像。
  - `COLOR_BGRA2RGBA`：将 BGRA 图像转换为 RGBA 图像。
  - `COLOR_RGBA2BGRA`：将 RGBA 图像转换为 BGRA 图像。
  - `COLOR_BGR2GRAY`：将 BGR 图像转换为 GRAY 图像。
  - `COLOR_RGB2GRAY`：将 RGB 图像转换为 GRAY 图像。
  - `COLOR_GRAY2BGR`：将 GRAY 图像转换为 BGR 图像。
  - `COLOR_GRAY2RGB`：将 GRAY 图像转换为 RGB 图像。
  - `COLOR_GRAY2BGRA`：将 GRAY 图像转换为 BGRA 图像。
  - `COLOR_GRAY2RGBA`：将 GRAY 图像转换为 RGBA 图像。
  - `COLOR_BGRA2GRAY`：将 BGRA 图像转换为 GRAY 图像。
  - `COLOR_RGBA2GRAY`：将 RGBA 图像转换为 GRAY 图像。

- **out**：表示算子输出 Tensor，需调用 `aclCreateTensor` 接口创建 `aclTensor` 类型的数据（数据存放在 Device 侧），输出 Tensor 的 dataType 支持 `UINT8`/`FLOAT`、Format 支持 `NHWC`、不支持非连续的 Tensor，同时 N 支持 1 或空、C 支持 1 或 3 或 4（1 表示输入 GRAY 图，3 表示输入 RGB 图，4 表示包含 alpha 通道），dataType、Format 需要和 self 一致。
- **workspaceSize**：返回用户需要在 Device 侧申请的 workspace 大小。
- **executor**：返回 op 执行器，包含了算子计算流程。

### 返回值

返回 `acldvppStatus` 状态码，具体请参见 6.2 acldvpp 返回码。

## acldvppConvertColor

### 参数说明

- **workspace**：需调用 `aclrtMalloc` 接口申请 Device 内存，内存大小为 `workspaceSize`，`aclrtMalloc` 接口输出的内存地址在此处传入。
- **workspaceSize**：与 `acldvppConvertColorGetWorkspaceSize` 接口获取的 `workspaceSize` 保持一致。
- **executor**：op 执行器，包含了算子计算流程，与 `acldvppConvertColorGetWorkspaceSize` 接口的 executor 保持一致。
- **stream**：指定执行任务的 Stream，可复用已创建的 Stream 节省资源或调用 `aclrtCreateStream` 接口创建 Stream，再作为入参在此处传入。

### 返回值

返回 `acldvppStatus` 状态码，具体请参见 6.2 acldvpp 返回码。

## 约束说明

- 支持图像分辨率范围在 `[6*4~4096*8192]`。
- 涉及 alpha 通道的转换模式，只支持 `UINT8` 类型输入和输出数据。
