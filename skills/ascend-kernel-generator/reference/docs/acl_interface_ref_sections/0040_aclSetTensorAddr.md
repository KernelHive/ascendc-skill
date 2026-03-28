## aclSetTensorAddr

## 函数功能

通过 `aclSetAclOpExecutorRepeatable` 使能 `aclOpExecutor` 可复用后，若输入或输出 Device 内存地址变更，需要刷新对应 `aclTensor` 中记录的 Device 内存地址。

## 函数原型

```c
aclnnStatus aclSetTensorAddr(aclOpExecutor *executor, const size_t index,
                             aclTensor *tensor, void *addr)
```

## 参数说明

| 参数名   | 输入/输出 | 说明                                                                 |
|----------|-----------|----------------------------------------------------------------------|
| executor | 输入      | 设置为复用状态的 `aclOpExecutor`                                     |
| index    | 输入      | 待刷新的 `aclTensor` 索引，取值范围是 `[0, tensor 的总数-1]`         |
| tensor   | 输入      | 待刷新的 `aclTensor` 指针                                            |
| addr     | 输入      | 需要刷新到指定 `aclTensor` 中的 Device 存储地址                      |

## 返回值说明

返回 0 表示成功，返回其他值表示失败，返回码列表参见“公共接口返回码”。

可能失败的原因：

- 返回 561103：`executor` 或 `tensor` 是空指针
- 返回 161002：`index` 取值越界
- 返回 161002：第一次执行一阶段接口 `aclxxXxxGetWorkspaceSize` 时传入的 `aclTensor` 是 `nullptr`，不再支持刷新地址

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
aclSetTensorAddr(executor, 0, tensor1, addr); // 刷新输入 tensorlist 中第 1 个 aclTensor 的 device 地址
aclSetTensorAddr(executor, 1, tensor2, addr); // 刷新输入 tensorlist 中第 2 个 aclTensor 的 device 地址
aclSetTensorAddr(executor, 2, tensor3, addr); // 刷新输入 aclTensor 的 device 地址
aclSetTensorAddr(executor, 3, output, addr);  // 刷新输出 aclTensor 的 device 地址

...

// 调用第 2 段接口
aclnnAddCustom(workspace, workspaceSize, executor, stream);

// 清理 executor
aclDestroyAclOpExecutor(executor);
```
