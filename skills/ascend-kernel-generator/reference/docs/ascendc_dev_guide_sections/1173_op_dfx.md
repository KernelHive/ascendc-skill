###### op_dfx

```markdown
> **注意**：本章接口为预留接口，后续有可能变更或废弃，不建议开发者使用，开发者无需关注。

## 接口列表

| 接口定义 | 功能说明 |
|----------|----------|
| `ToString(const std::string &str)` | 将std::string转换为ge::AscendString。 |
| `OpDfxGuard(const char *file, int line, OpLevel level, const char *funcName, const char *paramNamesIn, const char *paramNamesOut, const INPUT_TUPLE &&in, const OUTPUT_TUPLE &&out)` | DFX统计guard的构造函数。 |
| `OpDfxGuard(const char *file, int line, OpLevel level, const char *funcName)` | - |
| `OpDfxGuard(uint32_t id, const char *file, int line, OpLevel level, const char *funcName, const char *paramNames, const std::tuple<Args...> &t)` | - |
| `OpDfxGuard(uint64_t id, DfxProfilingType type)` | - |
| `OpDfxGuard()` | - |
| `ValidDfxName([[maybe_unused]]char const *a, [[maybe_unused]]char const *b)` | 判断a == b？ |
| `OpGetLogSequence()` | 获取日志序号。 |
| `GenSummaryItemId(const char *l2Name, const char *l0Name)` | 用L2和L0的接口名生成一个唯一的统计id。 |
| `GenSummaryItemId(const char *l2Name, const char *l0Name, const char *opType)` | 用L2和L0的接口名以及算子类型名生成一个唯一的统计id。 |
| `GenKernelLauncherId(const char *l0Name)` | 用L0接口名生成kernel launch的统计id。 |
| `OpProfilingSwitch()` | OpProfilingSwitch构造函数。 |
| `OpLogInfo()` | OpLogInfo构造函数。 |
| `OpLogInfo(const OpLogInfo &rhs)` | OpLogInfo拷贝构造函数。 |
| `InitLevelZero()` | 初始化L0统计信息。 |
| `InitLevelTwo()` | 初始化L2统计信息。 |
| `PtrCastTo(From *ptr)` | 将ptr转换为指定类型的指针。 |
| `PtrCastTo(const From *ptr)` | - |
| `GenOpTypeId(const char *opName)` | 以给定的算子名生成唯一的算子id。 |
| `IsDumpEnabled()` | 判断是否使能dump。 |
| `InitThreadLocalContext()` | 初始化线程局部上下文信息。 |
| `AddInputTensorToThreadLocalCtx(const aclTensor *const t)` | 线程局部上下文信息中增加input tensor信息。 |
| `AddInputTensorToThreadLocalCtx(aclTensor *const t)` | - |
| `AddInputTensorToThreadLocalCtx([[maybe_unused]] T &t)` | - |
| `AddInputTensorsToThreadLocalCtx(const std::tuple<Args...> &t)` | 线程局部上下文信息中增加多个input tensor信息。 |
| `AddInputTensorToThreadLocalCtx(const aclTensorList *const t)` | 线程局部上下文信息中增加input tensorList信息。 |
| `AddInputTensorToThreadLocalCtx(aclTensorList *const t)` | - |
| `AddOutputTensorToThreadLocalCtx(const aclTensor *const t)` | 线程局部上下文信息中增加output tensor信息。 |
| `AddOutputTensorToThreadLocalCtx(aclTensor *const t)` | - |
| `AddOutputTensorToThreadLocalCtx([[maybe_unused]] T &t)` | - |
| `AddOutputTensorsToThreadLocalCtx(const std::tuple<Args...> &t)` | 线程局部上下文信息中增加多个output tensor信息。 |
| `AddOutputTensorToThreadLocalCtx(const aclTensorList *const t)` | 线程局部上下文信息中增加output tensorList信息。 |
| `AddOutputTensorToThreadLocalCtx(aclTensorList *const t)` | - |
| `CreateDfxProfiler(const char *funcName)` | 用给定funcName创建统计器。 |
| `CreateDfxProfiler(uint32_t id)` | 用给定id创建统计器。 |
| `ToStr(const T &t, std::string &res, std::vector<std::string> &v, size_t &index)` | 将给定的参数转换为字符串。 |
| `StringToVec(const char *paramNames, std::vector<std::string> &v)` | 将给定字符串按逗号分隔。 |
| `StringToVecWithBrackets(const char *paramNames, std::vector<std::string> &v)` | 提取形如DFX_IN(aa, bb, cc)或DFX_OUT(aa, bb)字符串的子项。 |
| `SplitStringAndPrint(std::string &res)` | 将给定长字符串分为多个子字符串打印。 |
| `BuildParamString(const char *paramNames, const std::tuple<Args...> &t)` | 用给定的算子入参生成参数字符串。 |
| `BuildParamStringWithBrackets(const char *paramNames, const std::tuple<Args...> &t)` | - |
| `GenOpTypeId(const char *opName, const OP_RESOURCES &opResources)` | 用给定的算子名和二进制资源生成唯一的算子id。 |
| `GenOpTypeId(const char *opName, const OP_SOC_RESOURCES &opResources)` | 用给定的算子名和SOC二进制资源生成唯一的算子id。 |
| `GenInternalOpTypeId()` | 生成内部算子id。 |
| `aclnnStatus CheckPhase1Params(aclOpExecutor **executor, uint64_t *workspaceSize)` | 校验一阶段中，公共入参是否为nullptr。 |
```
