##### GetSubgraphNames

## 函数功能
获取算子IR信息中子图名称列表。

## 函数原型
```cpp
std::vector<std::string> GetSubgraphNames() const
graphStatus GetSubgraphNames(std::vector<AscendString> &names) const
```

## 须知
数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| names  | 输出      | 获取一个算子的子图名称列表。 |

## 返回值
graphStatus类型：
- GRAPH_SUCCESS：代表成功
- GRAPH_FAILED：代表失败

## 异常处理
无。

## 约束说明
无。
