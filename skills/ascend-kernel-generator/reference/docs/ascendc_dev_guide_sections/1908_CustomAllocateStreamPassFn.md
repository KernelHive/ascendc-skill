##### CustomAllocateStreamPassFn

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

## 功能说明

注册自定义的逻辑流分配Pass执行函数。

关于接口的详细使用方法请参见自定义Pass开发 > 使用自定义逻辑流分配Pass定制并发。

## 函数原型

```cpp
PassRegistrationData &CustomAllocateStreamPassFn(const CustomAllocateStreamPassFunc &allocate_stream_pass_fn)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| `allocate_stream_pass_fn` | 输入 | 自定义的逻辑流分配函数。详情请参见回调函数 `CustomAllocateStreamPassFunc`。 |

## 返回值说明

返回自身对象的引用。

## 约束说明

无。

## 回调函数 CustomAllocateStreamPassFunc

用户自定义并实现 `CustomAllocateStreamPassFunc` 类函数，即自定义的逻辑流分配函数。

```cpp
Status CustomAllocateStreamPassFunc(const ConstGraphPtr &graph, StreamPassContext &stream_context)
```

### 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| `graph` | 输入 | 待分配逻辑流的图 |
| `stream_context` | 输入 | 逻辑流分配上下文，可通过该上下文申请新stream id，设置节点的stream id等。详见15.2.3.31 StreamPassContext结构定义。 |

### 返回值

- **SUCCESS**：成功。
- 其他值：失败。
