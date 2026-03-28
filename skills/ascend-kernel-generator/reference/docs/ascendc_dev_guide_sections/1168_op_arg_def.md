###### op_arg_def

> 本章接口为预留接口，后续有可能变更或废弃，不建议开发者使用，开发者无需关注。

## 接口列表

| 接口定义 | 功能说明 |
|----------|----------|
| `VisitTupleElem(const F &func, const std::tuple<Ts...> &tp)` | 遍历OpArgBase的所有元素，调用func。 |
| `VisitTupleElemNoReturn(const F &func, const std::tuple<Ts...> &tp)` | 遍历OpArgBase的所有元素，调用func。 |
| `VisitTupleElemAt(size_t idx, const F &func, const std::tuple<Ts...> &tp)` | 对OpArgBase给定idx的元素，调用func。 |
| `OpArgBase(const std::tuple<T...> &arg)` | OpArgBase构造函数。 |
| `OpArgBase(std::tuple<T...> &&arg)` | OpArgBase构造函数。 |
| `OpArgBase()` | OpArgBase构造函数。 |
| `Size()` | 获取OpArgBase的元素个数。 |
| `VisitBy(const F &func)` | 遍历OpArgBase的所有元素，调用func。 |
| `VisitTupleElem(func, arg_)` | 遍历OpArgBase的所有元素，调用func。 |
| `VisitByNoReturn(const F &func)` | 对OpArgBase给定idx的元素，调用func。 |
| `VisitAt(size_t idx, const F &func)` | 对OpArgBase给定idx的元素，调用func。 |
| `OpArgTypeStr(int argType)` | 将argType转换为字符串。 |
| `ExtractOpArgType(const T &t, const Ts &...ts)` | 获取ts中类型为V的元素。 |
| `ExtractOpArgTypeTuple(const Tuple &t)` | 获取t中类型为V的元素。 |
| `OpArgValue()` | OpArgValue的构造函数。 |
| `OpArgValue(const aclTensor *value)` | 用给定的value构造OpArgValue并赋值。 |
| `OpArgValue(aclTensor *value)` | 用给定的value构造OpArgValue并赋值。 |
| `OpArgValue(const aclTensorList *value)` | 用给定的value构造OpArgValue并赋值。 |
| `OpArgValue(aclTensorList *value)` | 用给定的value构造OpArgValue并赋值。 |
| `OpArgValue(std::string *value)` | 用给定的value构造OpArgValue并赋值。 |
| `OpArgValue(const std::string *value)` | 用给定的value构造OpArgValue并赋值。 |
| `OpArgValue(std::string &value)` | 用给定的value构造OpArgValue并赋值。 |
| `OpArgValue(const std::string &value)` | 用给定的value构造OpArgValue并赋值。 |
| `OpArgValue(const char *value)` | 用给定的value构造OpArgValue并赋值。 |
| `OpArgValue(char *value)` | 用给定的value构造OpArgValue并赋值。 |
| `OpArgValue(std::vector<std::tuple<void *, const aclTensor *>> *value)` | 用给定的value构造OpArgValue并赋值。 |
| `OpArgValue(const std::vector<std::tuple<void*, const aclTensor*>> *value)` | 用给定的value构造OpArgValue并赋值。 |
| `OpArgValue(double value)` | 用给定的value构造OpArgValue并赋值。 |
| `OpArgValue(uint32_t value)` | 用给定的value构造OpArgValue并赋值。 |
| `OpArgValue(int32_t value)` | 用给定的value构造OpArgValue并赋值。 |
| `OpArgValue(float value)` | 用给定的value构造OpArgValue并赋值。 |
| `OpArgValue(const bool value)` | 用给定的value构造OpArgValue并赋值。 |
| `OpArgValue(const DataType value)` | 用给定的value构造OpArgValue并赋值。 |
| `OpArgValue(aclScalar *value)` | 用给定的value构造OpArgValue并赋值。 |
| `OpArgValue(const aclScalar *value)` | 用给定的value构造OpArgValue并赋值。 |
| `OpArgValue(aclIntArray *value)` | 用给定的value构造OpArgValue并赋值。 |
| `OpArgValue(const aclIntArray *value)` | 用给定的value构造OpArgValue并赋值。 |
| `OpArgValue(aclFloatArray *value)` | 用给定的value构造OpArgValue并赋值。 |
| `OpArgValue(const aclFloatArray *value)` | 用给定的value构造OpArgValue并赋值。 |
| `OpArgValue(aclBoolArray *value)` | 用给定的value构造OpArgValue并赋值。 |
| `OpArgValue(const aclBoolArray *value)` | 用给定的value构造OpArgValue并赋值。 |
| `OpArgValue(op::OpImplMode value)` | 用给定的value构造OpArgValue并赋值。 |
| `OpArgValue(const T &value)` | 用给定的value构造OpArgValue并赋值。 |
| `OpArg()` | OpArg的构造函数。 |
| `OpArgList()` | OpArgList的构造函数。 |
| `OpArgList(OpArg *args_, size_t count_)` | 用给定的args创建OpArgList。 |
| `OpArgContext()` | OpArgContext的构造函数。 |
| `GetOpArg(OpArgDef type)` | 用给定的type获取OpArgList。 |
| `ContainsOpArgType(OpArgDef type)` | 判断OpArgContext是否包含执行类型的OpArgList。 |
| `AppendOpWorkspaceArg(aclTensorList *tensorList)` | 将给定的TensorList作为workspace参数加入OpArgContext中。 |
| `AppendOpArg([[maybe_unused]] size_t idx, const aclTensor *tensor, OpArg *&currArg)` | 将给定的参数加入OpArgList中。 |
| `AppendOpArg([[maybe_unused]] size_t idx, aclTensor *tensor, OpArg *&currArg)` | 将给定的参数加入OpArgList中。 |
| `AppendOpArg([[maybe_unused]] size_t idx, const aclTensorList *tensorList, OpArg *&currArg)` | 将给定的参数加入OpArgList中。 |
| `AppendOpArg([[maybe_unused]] size_t idx, aclTensorList *tensorList, OpArg *&currArg)` | 将给定的参数加入OpArgList中。 |
| `AppendOpArg([[maybe_unused]] size_t idx, [[maybe_unused]] const std::nullptr_t tensor, OpArg *&currArg)` | 将给定的参数加入OpArgList中。 |
| `AppendOpArg([[maybe_unused]] size_t idx, const bool value, OpArg *&currArg)` | 将给定的参数加入OpArgList中。 |
| `AppendOpArg([[maybe_unused]] size_t idx, const DataType value, OpArg *&currArg)` | 将给定的参数加入OpArgList中。 |
| `AppendOpArg([[maybe_unused]] size_t idx, aclScalar *value, OpArg *&currArg)` | 将给定的参数加入OpArgList中。 |
| `AppendOpArg([[maybe_unused]] size_t idx, const aclScalar *value, OpArg *&currArg)` | 将给定的参数加入OpArgList中。 |
| `AppendOpArg([[maybe_unused]] size_t idx, aclIntArray *value, OpArg *&currArg)` | 将给定的参数加入OpArgList中。 |
| `AppendOpArg([[maybe_unused]] size_t idx, const aclIntArray *value, OpArg *&currArg)` | 将给定的参数加入OpArgList中。 |
| `AppendOpArg([[maybe_unused]] size_t idx, aclFloatArray *value, OpArg *&currArg)` | 将给定的参数加入OpArgList中。 |
| `AppendOpArg([[maybe_unused]] size_t idx, const aclFloatArray *value, OpArg *&currArg)` | 将给定的参数加入OpArgList中。 |
| `AppendOpArg([[maybe_unused]] size_t idx, aclBoolArray *value, OpArg *&currArg)` | 将给定的参数加入OpArgList中。 |
| `AppendOpArg([[maybe_unused]] size_t idx, const aclBoolArray *value, OpArg *&currArg)` | 将给定的参数加入OpArgList中。 |
| `AppendOpArg([[maybe_unused]] size_t idx, std::string *value, OpArg *&currArg)` | 将给定的参数加入OpArgList中。 |
| `AppendOpArg([[maybe_unused]] size_t idx, const std::string *value, OpArg *&currArg)` | 将给定的参数加入OpArgList中。 |
| `AppendOpArg([[maybe_unused]] size_t idx, std::string &value, OpArg *&currArg)` | 将给定的参数加入OpArgList中。 |
| `AppendOpArg([[maybe_unused]] size_t idx, const std::string &value, OpArg *&currArg)` | 将给定的参数加入OpArgList中。 |
| `AppendOpArg([[maybe_unused]] size_t idx, const char *value, OpArg *&currArg)` | 将给定的参数加入OpArgList中。 |
| `AppendOpArg([[maybe_unused]] size_t idx, char *value, OpArg *&currArg)` | 将给定的参数加入OpArgList中。 |
| `AppendOpArg([[maybe_unused]] size_t idx, double value, OpArg *&currArg)` | 将给定的参数加入OpArgList中。 |
| `AppendOpArg([[maybe_unused]] size_t idx, float value, OpArg *&currArg)` | 将给定的参数加入OpArgList中。 |
| `AppendOpArg([[maybe_unused]] size_t idx, int32_t value, OpArg *&currArg)` | 将给定的参数加入OpArgList中。 |
| `AppendOpArg([[maybe_unused]] size_t idx, uint32_t value, OpArg *&currArg)` | 将给定的参数加入OpArgList中。 |
| `AppendOpArg([[maybe_unused]] size_t idx, op::OpImplMode value, OpArg *&currArg)` | 将给定的参数加入OpArgList中。 |
| `AppendOpArg([[maybe_unused]] size_t idx, std::vector<std::tuple<void*, const aclTensor*>> &value, OpArg *&currArg)` | 将给定的参数加入OpArgList中。 |
| `AppendOpArg([[maybe_unused]] size_t idx, const std::vector<std::tuple<void*, const aclTensor*>> &value, OpArg *&currArg)` | 将给定的参数加入OpArgList中。 |
| `AppendOpArg([[maybe_unused]] size_t idx, T value, OpArg *&currArg)` | 将给定的参数加入OpArgList中。 |
| `OpArgContextSize(const T &t, const Ts &...ts)` | 用给定的参数计算OpArgContext的大小。 |
| `OpArgContextInit(OpArgContext &ctx, OpArg *&currArg, const T &t, const Ts &...ts)` | 用给定的参数初始化OpArgContext。 |
| `MakeOpArgContext(const Ts &...ts)` | 用给定的参数创建OpArgContext。 |
| `GetOpArgContext(const Ts &...ts)` | 用给定的参数获取OpArgContext。 |
| `DestroyOpArgContext(OpArgContext *ctx)` | 销毁指定的OpArgContext。 |
| `Allocated(size_t size)` | 申请size大小的内存。 |
| `DeAllocated(void *addr)` | 释放地址addr的内存。 |
