##### GetParseSubgraphPostFn

## 函数功能

根据算子类型，获取算子注册的子图中输入输出节点跟算子的输入输出的对应关系实现的函数对象。

## 函数原型

### 须知

`GetParseSubgraphPostFn()` 函数后续版本将废弃，建议使用 `GetParseSubgraphPostFn(ParseSubgraphFuncV2 &func)` 函数。

- `ParseSubgraphFunc GetParseSubgraphPostFn() const`

  该函数会返回 `ParseSubgraphFunc` 类型的函数对象，`ParseSubgraphFunc` 函数的声明如下：

  ```cpp
  using ParseSubgraphFunc = std::function<Status(const std::string &subgraph_name, const ge::Graph &graph)>
  ```

- `Status GetParseSubgraphPostFn(ParseSubgraphFuncV2 &func) const`

  该函数会返回 `ParseSubgraphFuncV2` 类型的函数对象，`ParseSubgraphFuncV2` 函数的声明如下：

  ```cpp
  using ParseSubgraphFuncV2 = std::function<Status(const ge::AscendString &subgraph_name, const ge::Graph &graph)>
  ```

## 参数说明

### `GetParseSubgraphPostFn()` 函数

无。

### `GetParseSubgraphPostFn(ParseSubgraphFuncV2 &func)` 函数

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| func | 输出 | 实现算子注册的子图中输入输出节点跟算子的输入输出对应关系的函数对象。 |

## 约束说明

无。
