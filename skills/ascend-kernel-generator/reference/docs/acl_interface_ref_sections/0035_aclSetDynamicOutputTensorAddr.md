## aclSetDynamicOutputTensorAddr

## 函数功能

通过 `aclSetAclOpExecutorRepeatable` 使能 `aclOpExecutor` 可复用后，若输出 Device 内存地址变更，需要刷新输出 `aclTensorList` 中记录的 Device 内存地址。

## 函数原型

```c
aclnnStatus aclSetDynamicOutputTensorAddr(
    aclOpExecutor *executor,
    size_t irIndex,
    const size_t relativeIndex,
    aclTensorList *tensors,
    void *addr
)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| executor | 输入 | 设置为复用状态的 `aclOpExecutor` |
| irIndex | 输入 | 待刷新的 `aclTensorList` 在算子 IR 原型定义中的索引，从 0 开始计数 |
| relativeIndex | 输入 | 待刷新的 `aclTensor` 在 `aclTensorList` 中的索引。如果 `aclTensorList` 有 N 个 Tensor，其取值范围为 [0, N-1] |
| tensors | 输入 | 待刷新的 `aclTensorList` 指针 |
| addr | 输入 | 需要刷新到指定 `aclTensor` 中的 Device 存储地址 |

## 返回值说明

返回 0 表示成功，返回其他值表示失败，返回码列表参见 3.39 公共接口返回码。

可能失败的原因：

- 返回 561103：`executor` 或 `tensors` 是空指针
- 返回 161002：`relativeIndex >= tensors` 里 tensor 的个数
- 返回 161002：`irIndex >` 算子原型输出参数的个数

## 约束与限制

无

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
aclTensor *list[] = {tensor3, tensor4};
auto tensorList = aclCreateTensorList(list, 2);
uint64_t workspaceSize = 0;
aclOpExecutor *executor;

// AddCustom 算子有两个输入（aclTensor），一个输出（aclTensorList）
// 调用第 1 段接口
aclnnAddCustomGetWorkspaceSize(tensor1, tensor2, tensorList, &workspaceSize, &executor);

// 设置 executor 为可复用
aclSetAclOpExecutorRepeatable(executor);
void *addr;
aclSetDynamicOutputTensorAddr(executor, 0, 0, tensorList, addr); // 刷新输出 tensorlist 中第 1 个 aclTensor 的 device 地址
aclSetDynamicOutputTensorAddr(executor, 0, 1, tensorList, addr); // 刷新输出 tensorlist 中第 2 个 aclTensor 的 device 地址

// 调用第 2 段接口
aclnnAddCustom(workspace, workspaceSize, executor, stream);

// 清理 executor
aclDestroyAclOpExecutor(executor);
```
