###### Inputs

## 功能说明
将 `void*` 指针的 vector 设置为 KernelContext 的 inputs。

## 函数原型
```cpp
ContextBuilder &Inputs(std::vector<void *> inputs)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| inputs | 输入 | 保存输入的 `void*` 指针 vector |

## 返回值说明
返回当前 ContextBuilder 的对象。

## 约束说明
无

## 调用示例
```cpp
PlatformInfo platformInfo;
auto contextBuilder = context_ascendc::ContextBuilder().Inputs({nullptr, reinterpret_cast<void*>(&platformInfo)});
```
