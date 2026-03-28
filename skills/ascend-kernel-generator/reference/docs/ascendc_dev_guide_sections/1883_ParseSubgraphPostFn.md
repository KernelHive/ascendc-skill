##### ParseSubgraphPostFn

## 函数功能

根据算子类型，注册算子的子图中输入输出节点跟算子的输入输出的对应关系函数实现。

## 函数原型

```cpp
OpRegistrationData &ParseSubgraphPostFn(const ParseSubgraphFunc &subgraph_post_fn)
OpRegistrationData &ParseSubgraphPostFn(const ParseSubgraphFuncV2 &subgraph_post_fn)
```

## 须知

数据类型为 string 的接口后续版本会废弃，建议使用数据类型为非 string 的接口。

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| subgraph_post_fn | 输入 | 子图中输入输出节点跟算子的输入输出的对应关系函数对象。详见回调函数 ParseSubgraphFuncV2。 |

## 约束说明

无。

## 回调函数 ParseSubgraphFuncV2

用户自定义并实现 ParseSubgraphFuncV2 函数，完成解析子图中输入输出节点跟算子的输入输出的对应关系功能，回调函数原型定义如下：

```cpp
Status ParseSubgraphFuncV2(const ge::AscendString &subgraph_name, const ge::Graph &graph)
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| subgraph_name | 输入 | 子图名字。 |
| graph | 输出 | 构造的子图。 |
