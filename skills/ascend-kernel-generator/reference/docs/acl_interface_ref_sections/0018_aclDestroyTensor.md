## aclDestroyTensor

## 函数功能

销毁通过 `aclCreateTensor` 接口创建的 `aclTensor`。

## 函数原型

```c
aclnnStatus aclDestroyTensor(const aclTensor *tensor)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| tensor | 输入 | 需要销毁的 tensor 指针。 |

## 返回值说明

返回 `0` 表示成功，返回其他值表示失败。返回码列表参见 **公共接口返回码**。

## 约束与限制

无

## 调用示例

接口调用请参考 `aclCreateTensor` 的调用示例。
