##### TPosition

Ascend C uses an abstract logical position (TPosition) to represent different levels of storage when managing physical memory at various levels, replacing the concept of on‑chip physical storage to hide the hardware architecture. The main TPosition types include: VECIN, VECOUT, VECCALC, A1, A2, B1, B2, C1, C2, CO1, and CO2. Among these, VECIN, VECCALC, and VECOUT are mainly used for vector programming, while A1, A2, B1, B2, C1, C2, CO1, and CO2 are used for matrix programming. You can refer to Section 5.4 Programming Paradigms for the basic concepts of TPosition and Table 15‑28 for the mapping relationship between TPosition and physical storage.

TPosition is defined as follows:

```cpp
enum class TPosition : uint8_t {
    GM,
    A1,
    A2,
    B1,
    B2,
    C1,
    C2,
    CO1,
    CO2,
    VECIN,
    VECOUT,
    VECCALC,
    LCM = VECCALC,
    SPM,
    SHM = SPM,
    TSCM,
    C2PIPE2GM,
    C2PIPE2LOCAL,
    MAX,
};
```

The specific meanings of the TPosition enumeration values are as follows:

| Enum Value      | Description |
|-----------------|-------------|
| GM              | Global Memory, corresponding to the external storage of the AI Core. |
| VECIN           | Used for vector computation; the location for storing input data when moving data into the Vector computation unit. |
| VECOUT          | Used for vector computation; the location for storing output data when moving results out of the Vector computation unit. |
| VECCALC         | Used for vector/matrix computation; used when temporary variables are needed during computation. |
| A1              | Used for matrix computation; stores the entire A matrix, analogous to the L2 cache in a CPU’s multi‑level cache. |
| B1              | Used for matrix computation; stores the entire B matrix, analogous to the L2 cache in a CPU’s multi‑level cache. |
| C1              | Used for matrix computation; stores the entire Bias matrix, analogous to the L2 cache in a CPU’s multi‑level cache. |
| A2              | Used for matrix computation; stores the subdivided small block of the A matrix, analogous to the L1 cache in a CPU’s multi‑level cache. |
| B2              | Used for matrix computation; stores the subdivided small block of the B matrix, analogous to the L1 cache in a CPU’s multi‑level cache. |
| C2              | Used for matrix computation; stores the subdivided small block of the Bias matrix, analogous to the L1 cache in a CPU’s multi‑level cache. |
| CO1             | Used for matrix computation; stores the small block of the result C matrix, understood as Cube Out. |
| CO2             | Used for matrix computation; stores the entire result C matrix, understood as Cube Out. |
| LCM             | Local Cache Memory, representing a temporarily shared Unified Buffer space; an alias for VECCALC, providing the same functionality. |
| SPM             | Used for temporarily storing Unified Buffer data when there is a risk of Unified Buffer memory overflow. |
| SHM             | Alias for SPM. |
| TSCM            | Temp Swap Cache Memory, used for temporarily swapping data to extra space for Matmul operations. |
| C2PIPE2GM       | Used for storing FIXPIPE quantization parameters. |
| C2PIPE2LOCAL    | Reserved parameter. Reserved for future functionality; developers do not need to pay attention to it for now. |
