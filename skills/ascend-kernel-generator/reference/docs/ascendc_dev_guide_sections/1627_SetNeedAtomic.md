##### SetNeedAtomic

## 函数功能

用于设置是否需要框架来完成 atomic clean 操作，保证在算子 kernel 执行前，输出内存已经被初始化。

atomic clean 是指在算子执行前对 GM（Global Memory）上输出内存进行初始化的过程，以确保累加、求最大值和求最小值等操作的正确性。

- **累加场景**：在执行累加操作前，需要将全局内存中的相关值清零。
- **求最大值**：在执行求最大值操作前，需要将全局内存中的相关值初始化为对应数据类型的最小值。
- **求最小值**：在执行求最小值操作前，需要将全局内存中的相关值初始化为对应数据类型的最大值。

算子可以选择自行执行初始化操作，或者通过设置此接口让框架通过自动插入清零算子等方式来完成初始化操作。使用框架进行初始化可以利用框架的优化能力，例如在图模式下集中分配清零地址，从而提高资源管理和分配的效率。

SetNeedAtomic 和 InitValue 接口配合使用，通过 InitValue 接口来配置初始化哪些输出和具体的初始化值。InitValue 接口请参考《Ascend C 算子开发指南》中的“API 参考 > Ascend C API > Host API”章节。

## 函数原型

```cpp
ge::graphStatus SetNeedAtomic(const bool atomic)
```

## 参数说明

| 参数   | 输入/输出 | 说明 |
|--------|-----------|------|
| atomic | 输入      | - true：需要做 atomic clean。<br>- false：不需要做 atomic clean。 |

## 返回值说明

设置成功时返回 `ge::GRAPH_SUCCESS`。

关于 graphStatus 的定义，请参见 15.2.3.55 ge::graphStatus。

## 约束说明

SetNeedAtomic 和 InitValue 接口配合使用，否则会出现初始化不生效的情况。

## 调用示例

```cpp
ge::graphStatus Tiling4XXX(TilingContext* context) {
    auto ret = context->SetNeedAtomic(true);
    // ...
}
```
