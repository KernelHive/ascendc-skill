##### Graph 构造函数和析构函数

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

## 功能说明

Graph 构造函数和析构函数。

## 函数原型

**须知**

数据类型为 string 的接口后续版本会废弃，建议使用数据类型为非 string 且传 name 参数的接口。默认无参的构造函数构造的是一个非法的 Graph 对象。

```cpp
explicit Graph(const std::string& name)
explicit Graph(const char *name)
Graph()
~Graph()
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| name | 输入 | Graph 名称，按照指定的名称构造 Graph。 |

## 返回值说明

Graph 构造函数返回 Graph 类型的对象。

## 约束说明

无
