## acldvppRgbToGrayscale

## 支持的产品型号

- Atlas A2 训练系列产品/Atlas A2 推理系列产品
- Atlas A3 训练系列产品/Atlas A3 推理系列产品

## 功能说明

算子功能：RGB图像转换成灰度图像。

## 函数原型

每个算子有两段接口，必须先调用`acldvppRgbToGrayscaleGetWorkspaceSize`接口获取入参并根据计算流程计算所需workspace大小，再调用`acldvppRgbToGrayscale`接口执行计算。两段式接口如下：

### 第一段接口

```c
acldvppStatus acldvppRgbToGrayscaleGetWorkspaceSize(const aclTensor *self, 
                                                     uint32_t outputChannelsNum, 
                                                     aclTensor *out, 
                                                     uint64_t *workspaceSize, 
                                                     aclOpExecutor **executor)
```

### 第二段接口

```c
acldvppStatus acldvppRgbToGrayscale(void *workspace, 
                                    uint64_t workspaceSize, 
                                    aclOpExecutor *executor, 
                                    aclrtStream stream)
```

## acldvppRgbToGrayscaleGetWorkspaceSize

### 参数说明

- **self**：表示算子输入Tensor，需调用`aclCreateTensor`接口创建`aclTensor`类型的数据（数据存放在Device侧），输入Tensor的dataType支持UINT8/FLOAT、Format支持NCHW/NHWC、不支持非连续的Tensor，同时N支持1或空、C支持3（3表示输入RGB图）
- **outputChannelsNum**：输出灰度图的通道数，取值可为1或3。当取值为3时，返回图像各通道的像素值将相同
- **out**：表示算子输出Tensor，需调用`aclCreateTensor`接口创建`aclTensor`类型的数据（数据存放在Device侧），输出Tensor的dataType支持UINT8/FLOAT、Format支持NCHW/NHWC、不支持非连续的Tensor，同时N只支持1、C支持1和3（1表示GRAY图，3表示RGB图、但RGB三个通道的像素值相同），dataType、Format需要和self一致
- **workspaceSize**：返回用户需要在Device侧申请的workspace大小
- **executor**：返回op执行器，包含了算子计算流程

### 返回值

返回`acldvppStatus`状态码，具体请参见6.2 acldvpp返回码。

## acldvppRgbToGrayscale

### 参数说明

- **workspace**：需调用`aclrtMalloc`接口申请Device内存，内存大小为`workspaceSize`，`aclrtMalloc`接口输出的内存地址在此处传入
- **workspaceSize**：与`acldvppRgbToGrayscaleGetWorkspaceSize`接口获取的`workspaceSize`保持一致
- **executor**：op执行器，包含了算子计算流程，与`acldvppRgbToGrayscaleGetWorkspaceSize`接口的`executor`保持一致
- **stream**：指定执行任务的Stream，可复用已创建的Stream节省资源或调用`aclrtCreateStream`接口创建Stream，再作为入参在此处传入

### 返回值

返回`acldvppStatus`状态码，具体请参见6.2 acldvpp返回码。

## 约束说明

- 支持图像分辨率范围在[6×4~4096×8192]
- 输入输出数据类型、宽高保持一致
- 强制校验输入格式为RGB图像，输出格式为Gray图像
