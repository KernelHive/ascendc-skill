## acldvppPosterize

```markdown
## 支持的产品型号

- Atlas A2 训练系列产品/Atlas A2 推理系列产品
- Atlas A3 训练系列产品/Atlas A3 推理系列产品

## 功能说明

算子功能：通过减少每个颜色通道的位数来生成图像。

## 函数原型

每个算子有两段接口，必须先调用"acldvppPosterizeGetWorkspaceSize"接口获取入参并根据计算流程计算所需workspace大小，再调用"acldvppPosterize"接口执行计算。

两段式接口如下：

### 第一段接口

```c
acldvppStatus acldvppPosterizeGetWorkspaceSize(const aclTensor *self, int32_t bits, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
```

### 第二段接口

```c
acldvppStatus acldvppPosterize(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
```

## acldvppPosterizeGetWorkspaceSize

### 参数说明

- **self**：表示算子输入Tensor，需调用aclCreateTensor接口创建aclTensor类型的数据（数据存放在Device侧），输入Tensor的dataType支持UINT8、Format支持NCHW/NHWC、不支持非连续的Tensor，同时N支持1或空、C支持1或3（1表示输入GRAY图，3表示输入RGB图）。
- **bits**：每个通道保留的位数，取值范围：[0, 8]。
- **out**：表示算子输出Tensor，需调用aclCreateTensor接口创建aclTensor类型的数据（数据存放在Device侧），输出Tensor的dataType支持UINT8、Format支持NCHW/NHWC、不支持非连续的Tensor，同时N只支持1或空、C支持1或3（1表示GRAY图，3表示RGB图），dataType、Format、Shape需要和self一致。
- **workspaceSize**：返回用户需要在Device侧申请的workspace大小。
- **executor**：返回op执行器，包含了算子计算流程。

### 返回值

返回acldvppStatus状态码，具体请参见6.2 acldvpp返回码。

## acldvppPosterize

### 参数说明

- **workspace**：需调用aclrtMalloc接口申请Device内存，内存大小为workspaceSize，aclrtMalloc接口输出的内存地址在此处传入。
- **workspaceSize**：与acldvppPosterizeGetWorkspaceSize接口获取的workspaceSize保持一致。
- **executor**：op执行器，包含了算子计算流程，与acldvppPosterizeGetWorkspaceSize接口的executor保持一致。
- **stream**：指定执行任务的Stream，可复用已创建的Stream节省资源或调用aclrtCreateStream接口创建Stream，再作为入参在此处传入。

### 返回值

返回acldvppStatus状态码，具体请参见6.2 acldvpp返回码。

## 约束说明

- 支持图像分辨率范围在[6*4~4096*8192]
```
