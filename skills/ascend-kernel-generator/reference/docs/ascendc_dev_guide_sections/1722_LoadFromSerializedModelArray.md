##### LoadFromSerializedModelArray

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

## 功能说明

从 ModelDef 的序列化数据中恢复 Graph。

## 函数原型

```cpp
graphStatus LoadFromSerializedModelArray(const void *serialized_model, size_t size)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| serialized_model | 输入 | ModelDef 序列化数据的指针。 |
| size | 输入 | ModelDef 序列化数据的长度。 |

## 返回值说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| - | graphStatus | GRAPH_SUCCESS(0)：成功。<br>其他值：失败。 |

## 约束说明

仅支持读取 air 格式的文件。
