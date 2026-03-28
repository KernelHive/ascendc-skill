##### Workspace

## 函数功能

设置 Workspace 内存指针，可以传多块 Workspace 地址。

## 函数原型

```cpp
OpTilingContextBuilder &Workspace(const gert::ContinuousVector *workspace)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| workspace | 输出 | ContinuousVector *结构，每一个元素对应一个 Workspace 指针。 |

## 返回值说明

OpTilingContextBuilder 对象本身，用于链式调用。

## 约束说明

- 在调用 Build 方法之前，必须调用该接口，否则构造出的 TilingContext 将包含未定义数据。
- 通过指针传入的参数（void*），其内存所有权归调用者所有；调用者必须确保指针在 ContextHolder 对象的生命周期内有效。
