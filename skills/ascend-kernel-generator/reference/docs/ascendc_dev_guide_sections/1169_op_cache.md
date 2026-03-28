###### op_cache

> 本章接口为预留接口，后续有可能变更或废弃，不建议开发者使用，开发者无需关注。

## 接口列表

| 接口定义 | 功能说明 |
|----------|----------|
| `GetCacheBuf()` | 获取aclnn cache保存需要的内存buf。 |
| `CheckCacheable()` | 判断当前host侧aclnn API是否支持aclnn cache。 |
| `AddSeperator()` | 在aclnn cache的hashkey计算依据中，添加一个分隔符。 |
| `AddParamToBuf(const T *addr, uint64_t size)` | 在aclnn cache的hashkey计算依据中，添加指定数据结构的关键信息。 |
| `AddParamToBuf(const aclTensor *tensor)` | 在aclnn cache的hashkey计算依据中，添加aclTensor参数。 |
| `AddParamToBuf(aclTensor *tensor)` | 在aclnn cache的hashkey计算依据中，添加aclTensor参数。 |
| `AddParamToBuf(const aclScalar *scalar)` | 在aclnn cache的hashkey计算依据中，添加aclScalar参数。 |
| `AddParamToBuf(const aclIntArray *value)` | 在aclnn cache的hashkey计算依据中，添加aclIntArray参数。 |
| `AddParamToBuf(const aclBoolArray *value)` | 在aclnn cache的hashkey计算依据中，添加aclBoolArray参数。 |
| `AddParamToBuf(const aclFloatArray *value)` | 在aclnn cache的hashkey计算依据中，添加aclFloatArray参数。 |
| `AddParamToBuf(const aclFp16Array *value)` | 在aclnn cache的hashkey计算依据中，添加aclFp16Array参数。 |
| `AddParamToBuf(const aclTensorList *tensors)` | 在aclnn cache的hashkey计算依据中，添加aclTensorList参数。 |
| `AddParamToBuf(const aclScalarList *scalars)` | 在aclnn cache的hashkey计算依据中，添加aclScalarList参数。 |
| `AddParamToBuf(aclTensorList *tensors)` | 在aclnn cache的hashkey计算依据中，添加aclTensorList参数。 |
| `AddParamToBuf(aclScalarList *scalars)` | 在aclnn cache的hashkey计算依据中，添加aclScalarList参数。 |
| `AddParamToBuf(const std::string &s)` | 在aclnn cache的hashkey计算依据中，添加std::string参数。 |
| `AddParamToBuf(const aclDataType dtype)` | 在aclnn cache的hashkey计算依据中，添加aclDataType参数。 |
| `AddParamToBuf(const char *c)` | 在aclnn cache的hashkey计算依据中，添加const char*参数。 |
| `AddParamToBuf(char *c)` | 在aclnn cache的hashkey计算依据中，添加char*参数。 |
| `AddParamToBuf()` | 在aclnn cache的hashkey计算依据中，添加参数。 |
| `AddParamToBuf(const T &value)` | 在aclnn cache的hashkey计算依据中，添加泛型参数。 |
| `AddOpConfigInfoToBuf()` | 在aclnn cache的hashkey计算依据中，添加算子的配置信息，如：确定性计算。 |
| `InitExecutorCacheThreadLocal()` | 初始化aclnn cache相关的线程变量。 |
| `GetFromCache(aclOpExecutor **executor, uint64_t *workspaceSize)` | 尝试获取匹配上的aclnn cache。 |
| `CaculateHashKey(const std::tuple<Args...> &t)` | 根据输入输出计算aclnn cache的hashkey。 |
| `GetFromCache(aclOpExecutor **executor, uint64_t *workspaceSize, const char *api, const INPUT_TUPLE &in, const OUTPUT_TUPLE &out)` | 尝试获取与指定输入输出匹配的aclnn cache。 |
| `OpCacheKey()` | OpCacheKey为aclnn cache的key数据结构，该函数为其构造函数。 |
| `OpCacheKey(uint8_t *buf_, size_t len_)` | OpCacheKey构造函数。 |
| `OpCacheKey(const OpCacheKey &key)` | OpCacheKey拷贝构造函数。 |
| `ToString()` | OpCacheKey的打印函数，将内容表示为字符串形式。 |
| `operator()` | OpCacheKey的判等函数。 |
| `SetOpCacheKey(OpCacheKey &key)` | 设置当前输入/输出对应的OpCacheKey。 |
| `AddrRule()` | 描述了aclnn cache中记录的aclTensor与host侧API外部入参间的对应关系，用于aclnn cache的地址刷新。该函数为其构造函数。 |
| `AddrRule(bool w, uint64_t offset, int inx, int64_t l2Offset)` | AddrRule构造函数。 |
| `OpExecCache()` | OpExecCache描述了一份aclnn cache信息，该函数为其构造函数。 |
| `InitOpCacheKey()` | 初始化aclnn cache的hashkey。 |
| `InitCacheData()` | 初始化aclnn cache需要的外部信息。 |
| `RecordAddrRule(const aclTensor *t, AddrRule &rule)` | 在生成aclnn cache过程中，生成一个记录aclTensor与host侧API外部入参间的对应关系的对象。 |
| `AddLaunchTensor(const aclTensor *t, size_t dataLen)` | 在生成aclnn cache过程中，针对某个aclTensor，生成cache记录。 |
| `AddLaunchData(size_t dataLen)` | 在aclnn cache所在的内存上，申请一块指定大小的空间，用于填写cache内容。 |
| `SetCacheBuf(void *buf)` | 设置aclnn cache所用的临时内存。 |
| `UpdateTensorAddr(void *workspaceAddr, const std::vector<void *> &tensors)` | 在使用aclnn cache时，首先将当前host侧API的实际输入地址刷新到cache中。 |
| `SetBlockDim(uint32_t blockDim)` | 在生成aclnn cache的过程中，记录cache对象使用的核数。 |
| `SetCacheTensorInfo(void *infoLists)` | 为了使aclnn cache支持profiling上报，收集aclTensor的属性信息。 |
| `GetCacheTensorInfo(int index)` | 为了使aclnn cache支持profiling上报，获取aclTensor的属性信息。 |
| `AddTensorRelation(const aclTensor *tensorOut, const aclTensor *tensorMiddle)` | 在aclnn cache中，记录两块aclTensor的内存等价关系。 |
| `NewLaunchCache(size_t *offset, size_t *cap, R runner)` | 生成一块新的cache内存。 |
| `OldCacheClear()` | 清理被淘汰的cache。 |
| `Finalize()` | 完成aclnn cache的生成工作，将可用标记位置位。 |
| `SetWorkspaceSize(uint64_t val)` | 在生成aclnn cache的过程中，记录cache对象使用的workspace大小。 |
| `GetWorkspaceSize()` | 在使用aclnn cache时，获取cache对象使用的workspace大小。 |
| `GetHash()` | 获取当前cache的hashkey。 |
| `GetOpCacheKey()` | 获取当前cache的hashkey。 |
| `MarkOpCacheInvalid()` | 将当前aclnn cache置为弃用状态。 |
| `IsOpCacheValid()` | 查询当前aclnn cache是否为弃用状态。 |
| `DoSummaryProfiling(int index)` | 在使用aclnn cache时，进行profiling上报。 |
| `RestoreThreadLocal(int index)` | 在使用aclnn cache时，重置当前的线程上下文。 |
| `Run(void *workspaceAddr, const aclrtStream stream, const std::vector<void *> &tensors)` | 通过aclnn cache运行host侧API。 |
| `CanUse()` | 设置aclnn cache的可用状态。 |
| `SetUse()` | 获取aclnn cache的可用状态。 |
| `GetShrinkList()` | 获取当前的aclnn cache淘汰列表。 |
| `GetStorageRelation()` | 获取aclnn cache中记录的aclTensor内存等价关系表。 |
| `OpCacheValue()` | OpCacheValue是aclnn cache在LRU（Least Recently Used）中的数据节点封装，该函数为其构造函数。 |
| `OpCacheValue(OpExecCache *cache, OpCacheKey &key)` | OpCacheValue构造函数。 |
| `OpCacheValue(const OpCacheValue &value)` | OpCacheValue拷贝构造函数。 |
| `OpCacheValue(OpCacheValue &&value)` | OpCacheValue移动构造函数。 |
| `GetOpExecCache(uint64_t hash)` | 通过hashkey获取一个aclnn cache。 |
| `GetOpExecCache(OpCacheKey &key)` | 通过hashkey获取一个aclnn cache。 |
| `AddOpExecCache(OpExecCache *exec)` | 向aclnn cache全局管理中，添加一个aclnn cache对象。 |
| `RemoveExecCache(OpExecCache *exec)` | 在aclnn cache全局管理中，移除一个aclnn cache对象。 |
