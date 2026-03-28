## acldvppAutoContrast

## 支持的产品型号
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
- Atlas A3 训练系列产品/Atlas A3 推理系列产品

## 功能说明
通过对每个通道像素进行重新映射来达到最大化对比度的效果，图像中最暗的像素会被映射为黑色，最亮的像素会被映射为白色。另外，用户还可以通过调整cutoff、ignore参数，控制对比度效果。

## 函数原型
每个算子有两段接口，必须先调用“acldvppAutoContrastGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“acldvppAutoContrast”接口执行计算。两段式接口如下：

### 第一段接口
```c
acldvppStatus acldvppAutoContrastGetWorkspaceSize(const aclTensor *self, const aclFloatArray *cutoff, const aclIntArray *ignore, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
```

### 第二段接口
```c
acldvppStatus acldvppAutoContrast(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
```

## acldvppAutoContrastGetWorkspaceSize

### 参数说明
- **self**：表示算子输入Tensor，需调用aclCreateTensor接口创建aclTensor类型的数据（数据存放在Device侧），输入Tensor的dataType支持UINT8/FLOAT、Format支持NCHW/NHWC、不支持非连续的Tensor，同时N支持1或空、C支持1或3（1表示输入数据是GRAY格式的图，3表示输入数据是按R、G、B顺序排布的图）。当数据类型为FLOAT时，期望其值的范围为[0, 1]。
- **cutoff**：表示输入图像直方图中需要剔除的最暗和最亮像素的百分比。需调用aclCreateFloatArray接口创建aclFloatArray类型的数据，长度为2，第一个数代表最暗、第二个数代表最亮，数据值必须在[0.0, 50.0)范围内。如果传入空指针，则两个百分比都设置为默认值：0.0。
- **ignore**：表示输入图像直方图中需要忽略的背景像素值。需调用aclCreateIntArray接口创建aclIntArray类型的数据，长度<=256，数据值必须在[0, 255]范围内。如果传入空指针，则默认不忽略像素值。
- **out**：表示算子输出Tensor，需调用aclCreateTensor接口创建aclTensor类型的数据（数据存放在Device侧），输出Tensor的dataType支持UINT8/FLOAT、不支持非连续的Tensor，dataType、Format、Shape需要和self一致。
- **workspaceSize**：返回用户需要在Device侧申请的workspace大小。
- **executor**：返回op执行器，包含了算子计算流程。

### 返回值
返回acldvppStatus状态码，具体请参见6.2 acldvpp返回码。

## acldvppAutoContrast

### 参数说明
- **workspace**：需调用aclrtMalloc接口申请Device内存，内存大小为workspaceSize，aclrtMalloc接口输出的内存地址在此处传入。
- **workspaceSize**：与acldvppAutoContrastGetWorkspaceSize接口获取的workspaceSize保持一致。
- **executor**：op执行器，包含了算子计算流程，与acldvppAutoContrastGetWorkspaceSize接口的executor保持一致。
- **stream**：指定执行任务的Stream，可复用已创建的Stream节省资源或调用aclrtCreateStream接口创建Stream，再作为入参在此处传入。

### 返回值
返回acldvppStatus状态码，具体请参见6.2 acldvpp返回码。

## 约束说明
- 支持图像分辨率范围在[6*4~4096*8192]。
- dataType为UINT8时，结果与Torchvision一致；dataType为FLOAT时，结果与Torchvision最大差异为±1/255。
