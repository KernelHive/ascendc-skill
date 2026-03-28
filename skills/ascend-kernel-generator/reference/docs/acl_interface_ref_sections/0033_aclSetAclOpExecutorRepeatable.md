## aclSetAclOpExecutorRepeatable

## 函数功能

使能 `aclOpExecutor` 为可复用状态。当用户想复用已有的 `aclOpExecutor` 时，必须在第一阶段接口 `aclxxXxxGetWorkspaceSize` 运行完成后，立即使用该接口使能复用，后续可多次调用第二阶段接口 `aclXxx` 进行算子执行。

`aclOpExecutor` 是框架定义的算子执行器，用于执行算子计算的容器，开发者无需关注其内部实现。

## 函数原型

```c
aclnnStatus aclSetAclOpExecutorRepeatable(aclOpExecutor *executor)
```

## 参数说明

| 参数名    | 输入/输出 | 说明                         |
|-----------|-----------|------------------------------|
| executor  | 输入      | 待设置复用的 `aclOpExecutor` |

## 返回值说明

返回 0 表示成功，返回其他值表示失败，返回码列表参见“3.39 公共接口返回码”。

### 可能失败的原因

- 返回 561103：`executor` 是空指针。

## 约束与限制

- 目前采用 AI CPU 和 AI Core 计算单元的算子支持使能 `aclOpExecutor` 可复用。
- 调用单算子 API 执行接口时，如下场景无法使能 `aclOpExecutor` 复用：
  - 如果使用了 HostToDevice、DeviceToDevice 拷贝相关的 L0 层 API，如 `CopyToNpu`、`CopyNpuToNpu`、`CopyToNpuSync` 等，不支持 `aclOpExecutor` 复用。
  - 如果使用了 L0 层 `ViewCopy` 接口，同时 `ViewCopy` 的源地址和目的地址相同时，不支持 `aclOpExecutor` 复用。
  - 关于 L0 层接口的具体介绍请参见《Ascend C 算子开发指南》中“API 参考 > Ascend C API > Host API > 单算子 API 执行相关接口”。
- 调用单算子 API 执行接口时，不允许算子 API 内部创建 Device Tensor，只允许使用外部传入的 Tensor。
- 设置成复用状态的 `aclOpExecutor` 在第二阶段接口执行完后不会对 `executor` 的资源进行清理，需要和 `aclDestroyAclOpExecutor` 配套使用清理资源。

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```c
// 创建输入和输出的 aclTensor 和 aclTensorList
std::vector<int64_t> shape = {1, 2, 3};
aclTensor tensor1 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
                                    nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
aclTensor tensor2 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
                                    nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
aclTensor tensor3 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
                                    nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
aclTensor output = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
                                   nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
aclTensor *list[] = {tensor1, tensor2};
auto tensorList = aclCreateTensorList(list, 2);
uint64_t workspaceSize = 0;
aclOpExecutor *executor;

// AddCustom 算子有两个输入（aclTensorList 和 aclTensor），一个输出（aclTensor）
// 调用第 1 段接口
aclnnAddCustomGetWorkspaceSize(tensorList, tensor3, output, &workspaceSize, &executor);

// 设置 executor 为可复用
aclSetAclOpExecutorRepeatable(executor);

void *addr;
aclSetDynamicInputTensorAddr(executor, 0, 0, tensorList, addr); // 刷新输入 tensorList 中第 1 个 aclTensor 的 device 地址
aclSetDynamicInputTensorAddr(executor, 0, 1, tensorList, addr); // 刷新输入 tensorList 中第 2 个 aclTensor 的 device 地址

// 调用第 2 段接口
aclnnAddCustom(workspace, workspaceSize, executor, stream);

// 清理 executor
aclDestroyAclOpExecutor(executor);
```
