##### VerifyAllAttr

## 函数功能

根据 `disableCommonVerifier` 值，校验 Operator 中的属性是否有效，校验 Operator 的输入输出是否有效。

## 函数原型

```cpp
graphStatus VerifyAllAttr(bool disable_common_verifier = false)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| `disable_common_verifier` | 输入 | 当为 `false` 时，只校验属性有效性；当为 `true` 时，增加校验 Operator 所有输入输出有效性。<br>默认值为 `false`。 |

## 返回值

`graphStatus` 类型：推导成功，返回 `GRAPH_SUCCESS`，否则，返回 `GRAPH_FAILED`。

## 异常处理

无。

## 约束说明

无。
