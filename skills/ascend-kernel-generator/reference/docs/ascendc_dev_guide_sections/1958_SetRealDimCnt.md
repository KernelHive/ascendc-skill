##### SetRealDimCnt

## 函数功能

向 TensorDesc 中设置 Tensor 的实际维度数目。

通过 `GetShape` 接口返回的 Shape 的维度可能存在补 1 的场景，因此可以通过该接口设置 Shape 的实际维度个数。

## 函数原型

```cpp
void SetRealDimCnt(const int64_t real_dim_cnt)
```

## 参数说明

| 参数名         | 输入/输出 | 描述                                   |
|----------------|-----------|----------------------------------------|
| real_dim_cnt   | 输入      | 需设置的 TensorDesc 的实际数据维度数目信息。 |

## 返回值

无。

## 异常处理

无。

## 约束说明

无。
