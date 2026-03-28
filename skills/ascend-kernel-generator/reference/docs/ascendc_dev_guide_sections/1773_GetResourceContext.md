##### GetResourceContext

## 函数功能

通过资源标识 key 来获取对应的资源上下文对象。

## 函数原型

```cpp
ResourceContext *GetResourceContext(const ge::AscendString &key)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| key    | 输入      | 资源的唯一标识。由资源类算子的 infershape 函数指定。 |

## 返回值

资源上下文对象。

基础定义如下，资源类算子可以基于此扩展：

```cpp
struct ResourceContext {
    virtual ~ResourceContext() {}
};
```

用于保存资源相关信息，如 shape、datatype 等。

## 约束说明

若使用 15.2.3.12.8 Create 接口创建 InferenceContext 时未传入 resource context 管理器指针，则该接口返回空指针，因此使用其返回值之前需要判空。
