### TfIdfVectorizer

## 功能
从输入 Tensor 中提取 n-grams，并将其保存为向量形式。

## 输入
**X**：输入 Tensor，数据类型支持 int32、int64、string（UTF-8），输入 shape 为 `[C]` 或 `[N, C]`，其中 N 为 batch，C 为序列长度。

## 属性
- **max_gram_length**：int，指定 n-grams 的最大长度。
- **max_skip_count**：int，指定生成 n-grams 时，在 X 中跳过的最大元素个数（词或字符）。若 `max_skip_count=1`，`min_gram_length=2`，`max_gram_length=3`，则可能生成 skip_count=0 和 skip_count=1 的 2-grams 和 3-grams。
- **min_gram_length**：int，指定 n-gram 的最小长度。若 `min_gram_length=2`，`max_gram_length=3`，输出中则可能包含 2-grams 和 3-grams。
- **mode**：string，权重标准，可以是 “TF”（term frequency）、“IDF”（inverse document frequency）和 “TFIDF”（TF and IDF）。
- **ngram_counts**：int 列表，不同长度 n-gram 在 pool 中的起始位置。
- **ngram_indexes**：int 列表，ngram-indexes 中的第 i 个元素表示第 i 个 n-gram 在输出 Tensor 中的坐标。
- **pool_int64s**：int 列表，表示从训练集学习到的 n-grams。
- **pool_strings**：string 列表，表示从训练集学习到的 n-grams。
- **weights**：float 列表，存储 pool 中每个 n-grams 的权重。

## 输出
**Y**：输出 Tensor，数据类型为 float。若输入 shape 为 `[C]`，则输出 shape 为 `[max(ngram_indexes) + 1]`；若输入 shape 为 `[N, C]`，则输出 shape 为 `[N, max(ngram_indexes) + 1]`。

## 限制与约束
`pool_int64s` 与 `pool_strings` 不可同时定义。

## 支持的 ONNX 版本
Opset v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
