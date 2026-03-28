## acldvppDecodeJpeg

## 支持的产品型号

- Atlas A2 训练系列产品/Atlas A2 推理系列产品
- Atlas A3 训练系列产品/Atlas A3 推理系列产品

## 功能说明

算子功能：将.jpg、.jpeg、.JPG、.JPEG图片文件解码为单通道（GRAY）或三通道（RGB）图像。

## 函数原型

每个算子有两段接口，必须先调用“acldvppDecodeJpegGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“acldvppDecodeJpeg”接口执行计算。两段式接口如下：

- **第一段接口：**
  ```c
  acldvppStatus acldvppDecodeJpegGetWorkspaceSize(const aclTensor* self, uint32_t channels, bool tryRecoverTruncated, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
  ```

- **第二段接口：**
  ```c
  acldvppStatus acldvppDecodeJpeg(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
  ```

## acldvppDecodeJpegGetWorkspaceSize

### 参数说明

- **self**：表示算子输入Tensor，需调用aclCreateTensor接口创建aclTensor类型的数据（数据存放在Device侧），输入Tensor的dataType支持UINT8、Format支持ND、不支持非连续的Tensor。
- **channels**：指定输出通道数量，1或者3，配置1输出GRAY图，配置3输出RGB图。
- **tryRecoverTruncated**：是否尝试解码损坏码流。该参数为预留参数，暂不支持。当前用户需配置为true。
- **out**：表示算子输出Tensor，需调用aclCreateTensor接口创建aclTensor类型的数据（数据存放在Device侧），输出Tensor的dataType支持UINT8、Format支持NCHW/NHWC、不支持非连续的Tensor。
- **workspaceSize**：返回用户需要在Device侧申请的workspace大小。
- **executor**：返回op执行器，包含了算子计算流程。

### 返回值

返回acldvppStatus状态码，具体请参见6.2 acldvpp返回码。

## acldvppDecodeJpeg

### 参数说明

- **workspace**：需调用aclrtMalloc接口申请Device内存，内存大小为workspaceSize，aclrtMalloc接口输出的内存地址在此处传入。
- **workspaceSize**：与acldvppDecodeJpegGetWorkspaceSize接口获取的workspaceSize保持一致。
- **executor**：op执行器，包含了算子计算流程，与acldvppDecodeJpegGetWorkspaceSize接口的executor保持一致。
- **stream**：指定执行任务的Stream，可复用已创建的Stream节省资源或调用aclrtCreateStream接口创建Stream，再作为入参在此处传入。

### 返回值

返回acldvppStatus状态码，具体请参见6.2 acldvpp返回码。

## 约束说明

- 支持分辨率范围在[6*4, 32768*32768]内的JPEG图像解码。
- 分辨率在[32*32, 16384*16384]范围内的图片会走专门的硬件加速解码。
- 其它场景（包含硬件解码失败的场景）使用开源软件libjpeg-turbo在AI CPU上解码。
- 在硬件解码场景下，本算子的解码结果与OpenCV 3.4.2版本保持一致，OpenCV 3.4.2版本中默认使用libjpeg-turbo 1.5.3版本。
