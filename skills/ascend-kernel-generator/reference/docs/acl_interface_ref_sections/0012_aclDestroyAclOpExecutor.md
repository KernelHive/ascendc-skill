## aclDestroyAclOpExecutor

## 函数功能

`aclOpExecutor` 是框架定义的算子执行器，用来执行算子计算的容器，开发者无需关注其内部实现，直接使用即可。

- 对于非复用状态的 `aclOpExecutor`，调用一阶段接口 `aclxxXxxGetworkspaceSize` 时框架会自动创建 `aclOpExecutor`，调用二阶段接口 `aclxxXxx` 时框架会自动释放 `aclOpExecutor`，无需手动调用本接口释放。
- 对于复用状态的 `aclOpExecutor`（调用 `aclSetAclOpExecutorRepeatable` 接口使能复用），算子执行器的管理由用户自行处理，因此 `aclOpExecutor` 的销毁需显式调用本接口手动销毁。

## 函数原型

```c
aclnnStatus aclDestroyAclOpExecutor(aclOpExecutor *executor)
```

## 参数说明

| 参数名    | 输入/输出 | 说明                     |
|-----------|-----------|--------------------------|
| executor  | 输入      | 待销毁的 `aclOpExecutor` |

## 返回值说明

返回 0 表示成功，返回其他值表示失败，返回码列表参见“公共接口返回码”。

可能失败的原因：

- 返回 561103：`executor` 是空指针。

## 约束与限制

本接口需与 `aclSetAclOpExecutorRepeatable` 接口配套使用，分别完成 `aclOpExecutor` 的复用与销毁。

## 调用示例

接口调用请参考 `aclSetAclOpExecutorRepeatable` 的调用示例。
