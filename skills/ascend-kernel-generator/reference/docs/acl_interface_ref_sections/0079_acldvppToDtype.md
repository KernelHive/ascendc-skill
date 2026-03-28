## acldvppToDtype

## 支持的产品型号

- Atlas A2 训练系列产品/Atlas A2 推理系列产品
- Atlas A3 训练系列产品/Atlas A3 推理系列产品

## 功能说明

转换数据类型。

## 函数原型

每个算子有两段接口，必须先调用“acldvppToDtypeGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“acldvppToDtype”接口执行计算。两段式接口如下：

- **第一段接口：**
  ```c
  acldvppStatus acldvppToDtypeGetWorkspaceSize(const aclTensor *self, aclDataType dtype, bool normalize, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
  ```

- **第二段接口：**
  ```c
  acldvppStatus acldvppToDtype(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
  ```

## acldvppToDtypeGetWorkspaceSize

### 参数说明

- **self**：表示算子输入Tensor，需调用aclCreateTensor接口创建aclTensor类型的数据（数据存放在Device侧），输入Tensor的dataType支持UINT8和FLOAT、Format支持NCHW/NHWC、不支持非连续的Tensor，同时N支持1和空、C支持1或3（1表示输入GRAY图，3表示输入RGB图）。
- **dtype**：转换目的的数据类型：支持UINT8，FLOAT。
- **normalize**：是否需要按照数据类型归一化。如果是false，则只是数据类型变化，值不变；如果是true，则会按照数据类型归一化。
- **out**：表示算子输出Tensor，需调用aclCreateTensor接口创建aclTensor类型的数据（数据存放在Device侧），输出Tensor的dataType支持UINT8和FLOAT、Format支持NHWC/NHWC、不支持非连续的Tensor，同时N支持1和空、C支持1或3（1表示输入GRAY图，3表示输入RGB图）。out的数据类型需要与self相同，out的C、H、W值需要与self相同。
- **workspaceSize**：返回用户需要在Device侧申请的workspace大小。
- **executor**：返回op执行器，包含了算子计算流程。

### 返回值

返回acldvppStatus状态码，具体请参见6.2 acldvpp返回码。

## acldvppToDtype

### 参数说明

- **workspace**：需调用aclrtMalloc接口申请Device内存，内存大小为workspaceSize，aclrtMalloc接口输出的内存地址在此处传入。
- **workspaceSize**：与acldvppToDtypeGetWorkspaceSize接口获取的workspaceSize保持一致。
- **executor**：op执行器，包含了算子计算流程，与acldvppToDtypeGetWorkspaceSize接口的executor保持一致。
- **stream**：指定执行任务的Stream，可复用已创建的Stream节省资源或调用aclrtCreateStream接口创建Stream，再作为入参在此处传入。

### 返回值

返回acldvppStatus状态码，具体请参见6.2 acldvpp返回码。

## 约束说明

支持图像分辨率范围在[6*4~4096*8192]。
