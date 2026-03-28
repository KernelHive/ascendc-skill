## acldvppEncodeJpeg

## 支持的产品型号
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
- Atlas A3 训练系列产品/Atlas A3 推理系列产品

## 功能说明
算子功能：将单通道（GRAY）或三通道（RGB）图像编码为JPEG图像。

## 函数原型
每个算子有两段接口，必须先调用 `acldvppEncodeJpegGetWorkspaceSize` 接口获取入参并根据计算流程计算所需 workspace 大小，再调用 `acldvppEncodeJpeg` 接口执行计算。两段式接口如下：

- **第一段接口：**
  ```c
  acldvppStatus acldvppEncodeJpegGetWorkspaceSize(const aclTensor* self, uint32_t quality, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
  ```

- **第二段接口：**
  ```c
  acldvppStatus acldvppEncodeJpeg(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
  ```

## acldvppEncodeJpegGetWorkspaceSize

### 参数说明
- **self**：表示算子输入 Tensor，需调用 `aclCreateTensor` 接口创建 `aclTensor` 类型的数据（数据存放在 Device 侧），输入 Tensor 的 dataType 支持 `UINT8`、Format 支持 `NCHW`/`NHWC`、不支持非连续的 Tensor，同时 N 支持 1 或空、C 支持 1 或 3（1 表示输入 GRAY 图，3 表示输入 RGB 图）。
- **quality**：指定编码质量，范围 [1, 100]，数值越小图片质量越差，但图片数据量小，占用内存少；数值越大图像质量越高，但图片数据量大，占用内存多。
- **out**：表示算子输出 Tensor，需调用 `aclCreateTensor` 接口创建 `aclTensor` 类型的数据（数据存放在 Device 侧），输出 Tensor 的 dataType 支持 `UINT8`、Format 支持 `ND`、不支持非连续的 Tensor。
- **workspaceSize**：返回用户需要在 Device 侧申请的 workspace 大小。
- **executor**：返回 op 执行器，包含了算子计算流程。

### 返回值
返回 `acldvppStatus` 状态码，具体请参见 6.2 acldvpp 返回码。

## acldvppEncodeJpeg

### 参数说明
- **workspace**：需调用 `aclrtMalloc` 接口申请 Device 内存，内存大小为 `workspaceSize`，`aclrtMalloc` 接口输出的内存地址在此处传入。
- **workspaceSize**：与 `acldvppEncodeJpegGetWorkspaceSize` 接口获取的 `workspaceSize` 保持一致。
- **executor**：op 执行器，包含了算子计算流程，与 `acldvppEncodeJpegGetWorkspaceSize` 接口的 `executor` 保持一致。
- **stream**：指定执行任务的 Stream，可复用已创建的 Stream 节省资源或调用 `aclrtCreateStream` 接口创建 Stream，再作为入参在此处传入。

### 返回值
返回 `acldvppStatus` 状态码，具体请参见 6.2 acldvpp 返回码。

## 约束说明
- 支持分辨率范围在 [32×32, 8192×8192] 内的 JPEG 图像编码。
- 原始图像的宽高需要是 2 对齐。
