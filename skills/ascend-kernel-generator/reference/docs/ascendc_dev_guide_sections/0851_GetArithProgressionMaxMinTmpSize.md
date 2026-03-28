##### GetArithProgressionMaxMinTmpSize

## 功能说明

用于获取 ArithProgression Tiling 参数：ArithProgression 接口能完成计算所需最大临时空间大小 max 和最小临时空间大小 min。

由于 ArithProgression 接口内部不需要用到临时空间，max 和 min 均返回 0。

## 函数原型

```cpp
void GetArithProgressionMaxMinTmpSize(uint32_t &maxValue, uint32_t &minValue)
```

## 参数说明

| 参数名   | 输入/输出 | 描述 |
|----------|-----------|------|
| maxValue | 输出      | ArithProgression 接口能完成计算所需最大临时空间大小。<br>**说明**：maxValue 仅作为参考值，有可能大于 Unified Buffer 剩余空间的大小，该场景下，开发者需要根据 Unified Buffer 剩余空间的大小来选取合适的临时空间大小。 |
| minValue | 输出      | ArithProgression 接口能完成计算所需最小临时空间大小。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

```cpp
uint32_t maxValue = 0;
uint32_t minValue = 0;
AscendC::GetArithProgressionMaxMinTmpSize(maxValue, minValue);
```
