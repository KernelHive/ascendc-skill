## aclSetRawTensorAddr

## 函数功能

刷新 aclTensor 中原始记录的 Device 内存地址，aclTensor 由 `aclCreateTensor` 接口创建。

通常情况下，若网络需要频繁复用 aclTensor（即保持 shape、format 等属性一致），可使用本接口刷新 aclTensor 原始的 Device 内存地址，达到复用的目的。

## 函数原型

```c
aclnnStatus aclSetRawTensorAddr(aclTensor *tensor, void *addr)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| tensor | 输入 | 待刷新的 aclTensor 指针 |
| addr   | 输入 | 需要刷新到指定 aclTensor 中的 Device 存储地址 |

## 返回值说明

返回 0 表示成功，返回其他值表示失败，返回码列表参见“公共接口返回码”。

可能失败的原因：

- 返回 161001：参数 tensor 为空指针

## 约束与限制

- 必须在一阶段接口 `aclxxXxxGetWorkspaceSize` 之前或者二阶段接口 `aclxxXxx` 之后使用，不支持在一阶段与二阶段接口之间使用
- 本接口可与 `aclGetRawTensorAddr` 接口配套使用，查看刷新后的结果是否符合预期

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```c
// 创建输入和输出张量 inputTensor 和 outputTensor
std::vector<int64_t> shape = {1, 2, 3};
aclTensor inputTensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
aclTensor outputTensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);

// 调用 Xxx 算子一、二阶段接口
auto ret = aclxxXxxGetWorkspaceSize(inputTensor, outputTensor, &workspaceSize, &executor);
ret = aclxxXxx(workspace, workspaceSize, executor, stream);
...

void *addr1;
void *addr2;
... // 申请 device 内存 addr1，addr2

ret = aclSetRawTensorAddr(inputTensor, addr1);  // 刷新输入张量 inputTensor 的 device 地址
ret = aclSetRawTensorAddr(outputTensor, addr2); // 刷新输出张量 outputTensor 的 device 地址
...

// 复用 inputTensor，outputTensor 后，调用 Yyy 算子 aclnn 一、二阶段接口
auto ret = aclnnYyyGetWorkspaceSize(inputTensor, outputTensor, &workspaceSize, &executor);
ret = aclnnYyy(workspace, workspaceSize, executor, stream);
...
```
