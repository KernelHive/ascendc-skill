###### Outputs

## 功能说明
将 `void*` 指针的 vector 设置为 KernelContext 的 output。

## 函数原型
```cpp
ContextBuilder &Outputs(std::vector<void *> outputs)
```

## 参数说明

| 参数     | 输入/输出 | 说明                         |
|----------|-----------|------------------------------|
| outputs  | 输入      | 保存输出的 void* 指针 vector |

## 返回值说明
当前 ContextBuilder 的对象。

## 约束说明
无

## 调用示例
```cpp
PlatformInfo platformInfo;
auto contextBuilder = context_ascendc::ContextBuilder().Outputs({nullptr, reinterpret_cast<void*>(&platformInfo)});
```
