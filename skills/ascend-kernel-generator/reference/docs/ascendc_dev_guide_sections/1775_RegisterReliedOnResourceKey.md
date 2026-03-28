##### RegisterReliedOnResourceKey

## 函数功能
注册依赖的资源。

一般由读类型的算子调用，如 stack pop。因读类型算子的 shape 依赖资源算子，调用该接口注册依赖的资源标识。若资源算子 shape 变化可触发读类型算子的重新推导。

## 函数原型
```cpp
graphStatus RegisterReliedOnResourceKey(const ge::AscendString &key)
```

## 参数说明

| 参数名 | 输入/输出 | 描述           |
|--------|-----------|----------------|
| key    | 输入      | 资源的唯一标识 |

## 返回值
graphStatus 类型：
- `GRAPH_SUCCESS`：代表成功
- `GRAPH_FAILED`：代表失败

## 约束说明
无。
