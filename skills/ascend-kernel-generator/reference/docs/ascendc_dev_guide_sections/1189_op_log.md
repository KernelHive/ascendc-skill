###### op_log

> **注意**：本章接口为预留接口，后续有可能变更或废弃，不建议开发者使用，开发者无需关注。

## 接口列表

| 接口定义 | 功能说明 |
|----------|----------|
| `GetTid()` | 获取线程id。 |
| `CheckLogLevel(int32_t moduleId, int32_t logLevel)` | 检查log等级。 |
| `DlogRecord(int32_t moduleId, int32_t level, const char *fmt, ...)` | 打印log。 |
| `ReportErrorMessageInner(const std::string &code, const char *fmt, ...)` | 报告内部错误消息。 |
| `ReportErrorMessage(const char *code, const char *fmt, Arguments &&... args)` | 报告错误消息。 |
| `GetLogApiInfo()` | 获取api信息。 |
| `GetFileName(const char *path)` | 获取文件名。 |
| `GetOpName()` | 获取算子名称。 |
