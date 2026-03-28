## acldvppImgToTensor

## 支持的产品型号
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
- Atlas A3 训练系列产品/Atlas A3 推理系列产品

## 功能说明
- **算子功能**：图像归一化，减均值、除标准差，但均值固定为0，标准差固定为255，仅用于将[0, 255]的UINT8图像归一化到[0.0, 1.0]的FLOAT图像。
- **计算公式**：
  ```
  out = (self - mean) / std
  ```

## 函数原型
每个算子有两段接口，必须先调用`acldvppImgToTensorGetWorkspaceSize`接口获取入参并根据计算流程计算所需workspace大小，再调用`acldvppImgToTensor`接口执行计算。两段式接口如下：

- **第一段接口**：
  ```c
  acldvppStatus acldvppImgToTensorGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
  ```

- **第二段接口**：
  ```c
  acldvppStatus acldvppImgToTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
  ```

## acldvppImgToTensorGetWorkspaceSize
### 参数说明
- **self**：表示算子输入Tensor，需调用`aclCreateTensor`接口创建`aclTensor`类型的数据（数据存放在Device侧），输入Tensor的dataType支持UINT8、Format支持NCHW/NHWC、不支持非连续的Tensor，同时N支持1或空、C支持1或3（1表示输入GRAY图，3表示输入RGB图）。
- **out**：表示算子输出Tensor，需调用`aclCreateTensor`接口创建`aclTensor`类型的数据（数据存放在Device侧），输出Tensor的dataType支持FLOAT、不支持非连续的Tensor，Format、Shape需要和self一致。
- **workspaceSize**：返回用户需要在Device侧申请的workspace大小。
- **executor**：返回op执行器，包含了算子计算流程。

### 返回值
返回`acldvppStatus`状态码，具体请参见6.2 acldvpp返回码。

## acldvppImgToTensor
### 参数说明
- **workspace**：需调用`aclrtMalloc`接口申请Device内存，内存大小为`workspaceSize`，`aclrtMalloc`接口输出的内存地址在此处传入。
- **workspaceSize**：与`acldvppImgToTensorGetWorkspaceSize`接口获取的`workspaceSize`保持一致。
- **executor**：op执行器，包含了算子计算流程，与`acldvppImgToTensorGetWorkspaceSize`接口的`executor`保持一致。
- **stream**：指定执行任务的Stream，可复用已创建的Stream节省资源或调用`aclrtCreateStream`接口创建Stream，再作为入参在此处传入。

### 返回值
返回`acldvppStatus`状态码，具体请参见6.2 acldvpp返回码。

## 约束说明
支持图像分辨率范围在[6*4~4096*8192]。
