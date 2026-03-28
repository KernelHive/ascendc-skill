## 支持 TensorFlow 算子清单

该算子规格仅适用于 TensorFlow 框架（TensorFlow 版本为 1.15 与 2.6.5）原生 IR 定义的网络模型，参数解释请参见 TensorFlow 官网。如果要查看基于 Ascend IR 定义的单算子信息，请参见 7 Ascend IR 算子规格说明。

| 支持的 TF 算子名称 | 算子分类     | 算子功能                                                                 |
|-------------------|-------------|--------------------------------------------------------------------------|
| Abs               | math_ops    | Computes the absolute value of a tensor.                                |
| AccumulateNV      | math_ops    | Returns the element-wise sum of a list of 2 tensors.                    |
| Acos              | math_ops    | Computes acos of x element-wise.                                        |
| Acosh             | math_ops    | Computes inverse hyperbolic cosine of x element-wise.                   |
| Add               | math_ops    | Returns x + y element-wise.                                             |
| AddN              | math_ops    | Add all input tensors element wise.                                     |
| AddV2             | math_ops    | Returns x + y element-wise.                                             |
| All               | math_ops    | Computes the "logical and" of elements across dimensions of a tensor.   |
| Any               | math_ops    | Computes the "logical or" of elements across dimensions of a tensor.    |
| ApproximateEqual  | math_ops    | Returns the truth value of abs(x-y) < tolerance element-wise.           |
| ArgMax            | math_ops    | Returns the index with the largest value across dimensions of a tensor. |
| ArgMin            | math_ops    | Returns the index with the smallest value across dimensions of a tensor.|
| Asin              | math_ops    | Computes asin of x element-wise.                                        |
| Asinh             | math_ops    | Computes inverse hyperbolic sine of x element-wise.                     |
| Atan              | math_ops    | Computes atan of x element-wise.                                        |
| Atan2             | math_ops    | Computes arctangent of y/x element-wise, respecting signs of the arguments. |
| Atanh             | math_ops    | Computes inverse hyperbolic tangent of x element-wise.                  |
| AvgPool           | nn_ops      | Performs average pooling on the input.                                  |
| Batch             | batch_ops   | -                                                                       |
| BatchMatMul       | math_ops    | Multiplies slices of two tensors in batches.                            |
| BatchToSpace      | array_ops   | BatchToSpace for 4-D tensors of type T.                                 |
| BatchToSpaceND    | array_ops   | BatchToSpace for N-D tensors of type T.                                 |
| BesselI0e         | math_ops    | Computes the Bessel i0e function of x element-wise.                     |
| BesselI1e         | math_ops    | Computes the Bessel i1e function of x element-wise.                     |
| Betainc           | math_ops    | Compute the regularized incomplete beta integral \(I_x(a, b)\).         |
| BiasAdd           | nn_ops      | Adds bias to value.                                                     |
| Bincount          | math_ops    | Counts the number of occurrences of each value in an integer array.     |
| BitwiseAnd        | bitwise_ops | -                                                                       |
| BitwiseOr         | bitwise_ops | -                                                                       |
| BitwiseXor        | bitwise_ops | -                                                                       |
| BroadcastTo       | array_ops   | Broadcast an array for a compatible shape.                              |
| Bucketize         | math_ops    | Bucketizes 'input' based on 'boundaries'.                               |
| Cast              | math_ops    | Cast x of type SrcT to y of DstT.                                       |
| Ceil              | math_ops    | Returns element-wise smallest integer not less than x.                  |
| CheckNumerics     | array_ops   | Checks a tensor for NaN and Inf values.                                 |
| Cholesky          | linalg_ops  | -                                                                       |
| CholeskyGrad      | linalg_ops  | -                                                                       |
| ClipByValue       | math_ops    | Clips tensor values to a specified min and max.                         |
| CompareAndBitpack | math_ops    | Compare values of input to threshold and pack resulting bits into a uint8. |
| Concat            | array_ops   | Concatenates tensors along one dimension.                               |
| ConcatV2          | array_ops   | -                                                                       |
| Const             | array_ops   | -                                                                       |
| ControlTrigger    | control_flow_ops | Does nothing.                                                       |
| Conv2D            | nn_ops      | Computes a 2-D convolution given 4-D input and filter tensors.          |
| Conv2DBackpropFilter | nn_ops   | Computes the gradients of convolution with respect to the filter.       |
| Conv2DBackpropInput | nn_ops    | Computes the gradients of convolution with respect to the input.        |
| Conv3D            | nn_ops      | Computes a 3D convolution given 5D "x" and "filter" tensor.             |
| Cos               | math_ops    | Computes cos of x element-wise.                                         |
| Cosh              | math_ops    | Computes hyperbolic cosine of x element-wise.                           |
| Cumprod           | math_ops    | Compute the cumulative product of the tensor x along axis.              |
| Cumsum            | math_ops    | Compute the cumulative sum of the tensor x along axis.                  |
| DataFormatDimMap  | nn_ops      | Returns the dimension index in the destination data format given the one in. |
| DataFormatVecPermute | nn_ops   | Returns the permuted vector/tensor in the destination data format given the. |
| DepthToSpace      | array_ops   | DepthToSpace for tensors of type T.                                     |
| DepthwiseConv2dNative | nn_ops | Computes a 2-D depthwise convolution given 4-D input and filter tensors. |
| DepthwiseConv2dNativeBackpropFilter | nn_ops | Computes the gradients of depthwise convolution with respect to the filter. |
| DepthwiseConv2dNativeBackpropInput | nn_ops | Computes the gradients of depthwise convolution with respect to the input. |
| Dequantize        | array_ops   | Dequantize the 'input' tensor into a float Tensor.                      |
| Diag              | array_ops   | Returns a diagonal tensor with a given diagonal values.                 |
| DiagPart          | array_ops   | Returns the diagonal part of the tensor.                                |
| Div               | math_ops    | Returns x / y element-wise.                                             |
| DivNoNan          | math_ops    | Returns 0 if the denominator is zero.                                   |
| Elu               | nn_ops      | Computes exponential linear: exp(features) - 1 if < 0, features otherwise. |
| Empty             | array_ops   | Creates a tensor with the given shape.                                  |
| Enter             | control_flow_ops | -                                                                   |
| Equal             | math_ops    | Returns the truth value of (x == y) element-wise.                       |
| Erf               | math_ops    | Computes the Gauss error function of x element-wise.                    |
| Erfc              | math_ops    | Computes the complementary error function of x element-wise.            |
| Exit              | control_flow_ops | -                                                                   |
| Exp               | math_ops    | Computes exponential of x element-wise.                                 |
| ExpandDims        | array_ops   | Inserts a dimension of 1 into a tensor's shape.                         |
| Expm1             | math_ops    | Computes exponential of x - 1 element-wise.                             |
| ExtractImagePatches | array_ops | Extract patches from images and put them in the "depth" output dimension. |
| FakeQuantWithMinMaxArgs | array_ops | Fake-quantize the 'inputs' tensor, type float to 'outputs' tensor of same type. |
| FakeQuantWithMinMaxVars | array_ops | Fake-quantize the 'inputs' tensor of type float via global float scalars min. |
| FakeQuantWithMinMaxVarsPerChannel | array_ops | Fake-quantize the 'inputs' tensor of type float and one of the shapes: [d],. |
| Fill              | array_ops   | Creates a tensor filled with a scalar value.                            |
| Floor             | math_ops    | Returns element-wise largest integer not greater than x.                |
| FloorDiv          | math_ops    | Returns x // y element-wise.                                            |
| FloorMod          | math_ops    | Returns element-wise remainder of division.                             |
| FractionalAvgPool | nn_ops      | Performs fractional average pooling on the input.                       |
| FractionalAvgPoolGrad | nn_ops  | -                                                                       |
| FractionalMaxPool | nn_ops      | Performs fractional max pooling on the input.                           |
| FractionalMaxPoolGrad | nn_ops  | -                                                                       |
| FusedBatchNorm    | nn_ops      | Batch normalization.                                                    |
| FusedBatchNormV2  | nn_ops      | Batch normalization.                                                    |
| Gather            | array_ops   | Gather slices from params according to indices.                         |
| GatherNd          | array_ops   | Gather slices from params into a Tensor with shape specified by indices. |
| GatherV2          | array_ops   | Gather slices from params axis according to indices.                    |
| Greater           | math_ops    | Returns the truth value of (x > y) element-wise.                        |
| GreaterEqual      | math_ops    | Returns the truth value of (x >= y) element-wise.                       |
| GuaranteeConst    | array_ops   | Gives a guarantee to the TF runtime that the input tensor is a constant. |
| HistogramFixedWidth | math_ops  | Return histogram of values.                                             |
| Identity          | array_ops   | Return a tensor with the same shape and contents as the input tensor or value. |
| IdentityN         | array_ops   | Returns a list of tensors with the same shapes and contents as the input. |
| Igamma            | math_ops    | Compute the lower regularized incomplete Gamma function P(a, x).        |
| Igammac           | math_ops    | Compute the upper regularized incomplete Gamma function Q(a, x).        |
| IgammaGradA       | math_ops    | -                                                                       |
| InplaceAdd        | array_ops   | Adds v into specified rows of x.                                        |
| InplaceSub        | array_ops   | Subtracts v into specified rows of x.                                   |
| InplaceUpdate     | array_ops   | Updates specified rows with values in v.                                |
| InTopK            | nn_ops      | Says whether the targets are in the top K predictions.                  |
| InTopKV2          | nn_ops      | Says whether the targets are in the top K predictions.                  |
| Inv               | math_ops    | Computes the reciprocal of x element-wise.                              |
| Invert            | bitwise_ops | -                                                                       |
| InvertPermutation | array_ops   | Computes the inverse permutation of a tensor.                           |
| IsVariableInitialized | state_ops | Checks whether a tensor has been initialized.                           |
| L2Loss            | nn_ops      | L2 Loss.                                                                |
| Less              | math_ops    | Returns the truth value of (x < y) element-wise.                        |
| LessEqual         | math_ops    | Returns the truth value of (x <= y) element-wise.                       |
| LinSpace          | math_ops    | Generates values in an interval.                                        |
| ListDiff          | array_ops   | -                                                                       |
| Log               | math_ops    | Computes natural logarithm of x element-wise.                           |
| Log1p             | math_ops    | Computes natural logarithm of (1 + x) element-wise.                     |
| LogicalAnd        | math_ops    | Returns the truth value of x AND y element-wise.                        |
| LogicalNot        | math_ops    | Returns the truth value of NOT x element-wise.                          |
| LogicalOr         | math_ops    | Returns the truth value of x OR y element-wise.                         |
| LogMatrixDeterminant | linalg_ops | -                                                                       |
| LogSoftmax        | nn_ops      | Computes log softmax activations.                                       |
| LoopCond          | control_flow_ops | Forwards the input to the output.                                   |
| LowerBound        | array_ops   | -                                                                       |
| LRN               | nn_ops      | Local Response Normalization.                                           |
| MatMul            | math_ops    | Multiply the matrix "a" by the matrix "b".                              |
| MatrixBandPart    | array_ops   | Copy a tensor setting everything outside a central band in each innermost matrix. |
| MatrixDeterminant | linalg_ops  | -                                                                       |
| MatrixDiag        | array_ops   | Returns a batched diagonal tensor with a given batched diagonal values. |
| MatrixDiagPart    | array_ops   | Returns the batched diagonal part of a batched tensor.                  |
| MatrixInverse     | linalg_ops  | -                                                                       |
| MatrixSetDiag     | array_ops   | Returns a batched matrix tensor with new batched diagonal values.       |
| MatrixSolve       | linalg_ops  | -                                                                       |
| MatrixSolveLs     | linalg_ops  | -                                                                       |
| MatrixTriangularSolve | linalg_ops | -                                                                       |
| Max               | math_ops    | Computes the maximum of elements across dimensions of a tensor.         |
| Maximum           | math_ops    | Returns the max of x and y.                                             |
| MaxPool           | nn_ops      | Performs max pooling on the input.                                      |
| MaxPoolV2         | nn_ops      | Performs max pooling on the input.                                      |
| MaxPool3D         | nn_ops      | Performs 3D max pooling on the input.                                   |
| MaxPoolWithArgmax | nn_ops      | Performs max pooling on the input and outputs both max values and indices. |
| Mean              | math_ops    | Computes the mean of elements across dimensions of a tensor.            |
| Merge             | control_flow_ops | Forwards the value of an available tensor from inputs to output.    |
| Min               | math_ops    | Computes the minimum of elements across dimensions of a tensor.         |
| Minimum           | math_ops    | Returns the min of x and y                                              |
| MirrorPad         | array_ops   | Pads a tensor with mirrored values.                                     |
| MirrorPadGrad     | array_ops   | -                                                                       |
| Mod               | math_ops    | Returns element-wise remainder of division.                             |
| Mul               | math_ops    | -                                                                       |
| Multinomial       | random_ops  | Draws samples from a multinomial distribution.                          |
| Neg               | math_ops    | -                                                                       |
| NextIteration     | control_flow_ops | Makes its input available to the next iteration.                    |
| NoOp              | no_op       | Does nothing.                                                           |
| NotEqual          | math_ops    | Returns the truth value of (x != y) element-wise.                       |
| NthElement        | nn_ops      | Finds values of the n-th order statistic for the last dimension.        |
| OneHot            | array_ops   | Returns a one-hot tensor.                                               |
| OnesLike          | array_ops   | Returns a tensor of ones with the same shape and type as x.             |
| Pack              | array_ops   | -                                                                       |
| Pad               | array_ops   | -                                                                       |
| ParallelConcat    | array_ops   | -                                                                       |
| ParameterizedTruncatedNormal | random_ops | Outputs random values from a normal distribution.               |
| Placeholder       | array_ops   | -                                                                       |
| PlaceholderWithDefault | array_ops | -                                                                   |
| PopulationCount   | bitwise_ops | -                                                                       |
| Pow               | math_ops    | Computes the power of one value to another.                             |
| PreventGradient   | array_ops   | -                                                                       |
| Prod              | math_ops    | Computes the product of elements across dimensions of a tensor.         |
| Qr                | linalg_ops  | -                                                                       |
| RandomGamma       | random_ops  | Outputs random values from the Gamma distribution(s) described by alpha. |
| RandomGammaGrad   | random_ops  | -                                                                       |
| RandomShuffle     | random_ops  | Randomly shuffles a tensor along its first dimension.                   |
| RandomStandardNormal | random_ops | -                                                                   |
| RandomUniform     | random_ops  | Outputs random values from a uniform distribution.                      |
| Range             | math_ops    | Creates a sequence of numbers.                                          |
| RandomUniformInt  | random_ops  | Outputs random integers from a uniform distribution.                    |
| Rank              | array_ops   | Returns the rank of a tensor.                                           |
| ReadVariableOp    | resource_variable_ops | -                                                                 |
| RealDiv           | math_ops    | Returns x / y element-wise for real types.                              |
| Reciprocal        | math_ops    | Computes the reciprocal of x element-wise.                              |
| RefEnter          | control_flow_ops | -                                                                   |
| RefExit           | control_flow_ops | -                                                                   |
| RefMerge          | control_flow_ops | -                                                                   |
| RefNextIteration  | control_flow_ops | Makes its input available to the next iteration.                    |
| RefSwitch         | control_flow_ops | Forwards the ref tensor data to the output port determined by pred. |
| Relu              | nn_ops      | Computes rectified linear: max(features, 0).                           |
| Relu6             | nn_ops      | Computes rectified linear 6: min(max(features, 0), 6).                 |
| Reshape           | array_ops   | Reshapes a tensor.                                                      |
| ReverseSequence   | array_ops   | Reverses variable length slices.                                        |
| ReverseV2         | array_ops   | -                                                                       |
| RightShift        | bitwise_ops | -                                                                       |
| Rint              | math_ops    | Returns element-wise integer closest to x.                              |
| Round             | math_ops    | Rounds the values of a tensor to the nearest integer, element-wise.     |
| Rsqrt             | math_ops    | Computes reciprocal of square root of x element-wise.                   |
| SegmentMax        | math_ops    | Computes the maximum along segments of a tensor.                        |
| Select            | math_ops    | -                                                                       |
| Selu              | nn_ops      | Computes scaled exponential linear: scale * alpha * (exp(features) - 1). |
| Shape             | array_ops   | Returns the shape of a tensor.                                          |
| ShapeN            | array_ops   | Returns shape of tensors.                                               |
| Sigmoid           | math_ops    | Computes sigmoid of x element-wise.                                     |
| Sign              | math_ops    | Returns an element-wise indication of the sign of a number.             |
| Sin               | math_ops    | Computes sin of x element-wise.                                         |
| Sinh              | math_ops    | Computes hyperbolic sine of x element-wise.                             |
| Size              | array_ops   | Returns the size of a tensor.                                           |
| Slice             | array_ops   | Return a slice from 'input'.                                            |
| Snapshot          | array_ops   | Returns a copy of the input tensor.                                     |
| Softmax           | nn_ops      | Computes softmax activations.                                           |
| Softplus          | nn_ops      | Computes softplus: log(exp(features) + 1).                              |
| Softsign          | nn_ops      | Computes softsign: features / (abs(features) + 1).                      |
| SpaceToBatch      | array_ops   | SpaceToBatch for 4-D tensors of type T.                                 |
| SpaceToBatchND    | array_ops   | SpaceToBatch for N-D tensors of type T.                                 |
| SpaceToDepth      | array_ops   | SpaceToDepth for tensors of type T.                                     |
| Split             | array_ops   | Splits a tensor into num_split tensors along one dimension.             |
| SplitV            | array_ops   | Splits a tensor into num_split tensors along one dimension.             |
| Sqrt              | math_ops    | Computes square root of x element-wise.                                 |
| Square            | math_ops    | Computes square of x element-wise.                                      |
| SquaredDifference | math_ops    | Returns (x - y)(x - y) element-wise.                                    |
| Squeeze           | array_ops   | Removes dimensions of size 1 from the shape of a tensor.                |
| StatelessMultinomial | stateless_random_ops | -                                                             |
| StopGradient      | array_ops   | Stops gradient computation.                                             |
| StridedSlice      | array_ops   | Return a strided slice from input.                                      |
| Sub               | math_ops    | -                                                                       |
| Sum               | math_ops    | Computes the sum of elements across dimensions of a tensor.             |
| Svd               | linalg_ops  | -                                                                       |
| Switch            | control_flow_ops | Forwards data to the output port determined by pred.                |
| Tan               | math_ops    | Computes tan of x element-wise.                                         |
| Tanh              | math_ops    | Computes hyperbolic tangent of x element-wise.                          |
| Tile              | array_ops   | Constructs a tensor by tiling a given tensor.                           |
| TopK              | nn_ops      | Finds values and indices of the k largest elements for the last dimension. |
| TopKV2            | nn_ops      | -                                                                       |
| Transpose         | array_ops   | Shuffle dimensions of x according to a permutation.                     |
| TruncateDiv       | math_ops    | Returns x / y element-wise for integer types.                           |
| TruncatedNormal   | random_ops  | Outputs random values from a truncated normal distribution.             |
| TruncateMod       | math_ops    | Returns element-wise remainder of division.                             |
| Unbatch           | batch_ops   | -                                                                       |
| UnbatchGrad       | batch_ops   | -                                                                       |
| Unique            | array_ops   | Finds unique elements in a 1-D tensor.                                  |
| UniqueWithCounts  | array_ops   | Finds unique elements in a 1-D tensor.                                  |
| Unpack            | array_ops   | -                                                                       |
| UnravelIndex      | array_ops   | Converts a flat index or array of flat indices into a tuple of.         |
| UnsortedSegmentMin | math_ops   | Computes the minimum along segments of a tensor.                        |
| UnsortedSegmentProd | math_ops  | Computes the product along segments of a tensor.                        |
| UnsortedSegmentSum | math_ops   | Computes the sum along segments of a tensor.                            |
| UpperBound        | array_ops   | -                                                                       |
| Variable          | state_ops   | Holds state in the form of a tensor that persists across steps.         |
| Where             | array_ops   | Returns locations of nonzero / true values in a tensor.                 |
| Xdivy             | math_ops    | Returns 0 if x == 0, and x / y otherwise, elementwise.                  |
| Xlogy             | math_ops    | Returns 0 if x == 0, and x * log(y) otherwise, elementwise.             |
| ZerosLike         | array_ops   | Returns a tensor of zeros with the same shape and type as x.            |
| Zeta              | math_ops    | Compute the Hurwitz zeta function \((x, q)\).                           |
| _Retval           | function_ops | -                                                                       |
| LeakyRelu         | nn_ops      | -                                                                       |
| FusedBatchNormV3  | nn_ops/mkl_nn_ops | -                                                                 |
