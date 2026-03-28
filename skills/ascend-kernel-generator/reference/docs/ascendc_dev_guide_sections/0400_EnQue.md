###### EnQue

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | × |
| Atlas 训练系列产品 | √ |

## 功能说明

将Tensor push到队列。

## 函数原型

```cpp
template <typename T>
__aicore__ inline bool EnQue(const LocalTensor<T>& tensor)
```

## 参数说明

### 模板参数说明

| 参数名 | 说明 |
|--------|------|
| T | Tensor的数据类型 |

### 参数说明

| 参数名 | 输入/输出 | 含义 |
|--------|-----------|------|
| tensor | 输入 | 指定的Tensor |

## 约束说明

无

## 返回值说明

- `true` - 表示Tensor加入Queue成功
- `false` - 表示Queue已满，入队失败

## 调用示例

```cpp
// 接口: EnQue Tensor
AscendC::TPipe pipe;
AscendC::TQueBind<AscendC::TPosition::VECOUT, AscendC::TPosition::GM, 4> que;
int num = 4;
int len = 1024;
pipe.InitBuffer(que, num, len);
AscendC::LocalTensor<half> tensor1 = que.AllocTensor<half>();
que.EnQue(tensor1); // 将tensor加入VECOUT的Queue中
```
