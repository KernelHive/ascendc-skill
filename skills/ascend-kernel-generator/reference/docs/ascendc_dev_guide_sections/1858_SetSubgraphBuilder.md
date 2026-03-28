##### SetSubgraphBuilder

## 函数功能
设置指定子图构建的函数对象。

## 函数原型
```cpp
void SetSubgraphBuilder(const char_t *ir_name, uint32_t index, const SubgraphBuilder &builder)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| ir_name | 输入 | 子图名称 |
| index | 输入 | 动态个数子图场景（子图数量不固定），标识子图的序号 |
| builder | 输入 | 子图构建的函数对象 |

## 返回值
无

## 异常处理
无

## 约束说明
无
