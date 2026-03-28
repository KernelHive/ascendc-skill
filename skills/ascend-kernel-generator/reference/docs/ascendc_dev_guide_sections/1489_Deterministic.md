##### Deterministic

## 函数功能
设置是否开启确定性计算。

## 函数原型
```cpp
OpTilingContextBuilder &Deterministic(int32_t deterministic)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|------------|------|
| deterministic | 输入 | 是否开启确定性计算（1为开启，0为关闭）。 |

## 返回值说明
返回 OpTilingContextBuilder 对象本身，用于链式调用。

## 约束说明
- 在调用 Build 方法之前，必须调用本接口，否则构造出的 TilingContext 将包含未定义数据。
- 通过指针传入的参数（void*），其内存所有权归调用者所有；调用者必须确保指针在 ContextHolder 对象的生命周期内有效。
