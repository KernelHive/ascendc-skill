## aclDestroyBoolArray

## 函数功能

销毁通过 `aclCreateBoolArray` 接口创建的 `aclBoolArray`。

## 函数原型

```c
aclnnStatus aclDestroyBoolArray(const aclBoolArray *array)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| array  | 输入      | 需要销毁的 `aclBoolArray` |

## 返回值说明

返回 `0` 表示成功，返回其他值表示失败。返回码列表参见 **3.39 公共接口返回码**。

## 约束与限制

无

## 调用示例

接口调用请参考 **3.2 aclCreateBoolArray** 的调用示例。
