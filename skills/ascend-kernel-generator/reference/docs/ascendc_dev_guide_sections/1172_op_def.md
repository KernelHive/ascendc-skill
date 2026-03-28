###### op_def

```markdown
> **注意**：本章接口为预留接口，后续有可能变更或废弃，不建议开发者使用，开发者无需关注。

## 接口列表

| 接口定义 | 功能说明 |
|----------|----------|
| `ToOpImplMode(const std::string &implModeStr)` | 将implModeStr字符串转换为OpImplMode |
| `ToString(OpImplMode implMode)` | 将OpImplMode转换为字符串 |
| `ImplModeToString(OpImplMode implMode)` | 将implModeStr字符串转换为OpImplMode |
| `Add(uint32_t &id, const char *opName)` | 将opName注册到OpTypeDict中，返回算子的id |
| `ToOpType(const std::string &opName)` | 用opName从OpTypeDict中查找算子的id |
| `ToString(uint32_t opType)` | 用算子的id从OpTypeDict中查找算子名 |
| `GetAllOpTypeSize()` | 获取OpTypeDict中注册算子的个数 |
| `ToOpTypeByConfigJson(const std::string &op_config_json)` | 用算子json文件名查找算子的id |
| `UpdateConfigJsonPath(uint32_t opType, const std::string &opFile)` | 用给定的算子的id更新算子的json文件名 |
| `ReadFile2String(const char *filename, std::string &content)` | 读取filename到content中 |
| `ReadDirBySuffix(const std::string &dir, const std::string &suffix, std::vector<std::string> &paths)` | 获取dir目录下所有后缀为suffix的文件 |
| `ToIndex(OpImplMode implMode)` | 获取OpImplMode的index |
| `ToIndexChar(OpImplMode implMode)` | 获取OpImplMode字符形式的index |
| `GetOpConfigJsonFileName(uint32_t opType)` | 根据算子id获取Config Json文件名 |
```
