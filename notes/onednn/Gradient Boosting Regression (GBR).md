# Gradient Boosting Regression (GBR) Performance Model

A **Gradient Boosting Regression (GBR) performance model** is a machine-learning model that uses a collection of decision trees to predict a continuous performance metric, such as:

- Latency
- Throughput
- Execution time
- Power consumption
- Memory usage

> **Note:** GBR usually means **Gradient Boosting Regression**, not “Gradient Boosting Reduction.”

## Basic Idea

Suppose we want to predict the latency of an LLM kernel.

### Input Features

```text
M, N, K
data_type
weight_bits
batch_size
GPU_frequency
memory_bandwidth
kernel_type
```

The GBR model processes these features and produces a prediction:

```text
Predicted latency = 1.35 ms
```

GBR builds multiple decision trees sequentially. Each new tree attempts to correct the prediction errors made by the previous trees.

The final prediction can be expressed as:

$$
F(x) = F_0(x) + \eta T_1(x) + \eta T_2(x) + \cdots + \eta T_n(x)
$$

Where:

- $F_0(x)$ is the initial prediction.
- $T_i(x)$ is the prediction from decision tree $i$.
- $\eta$ is the learning rate.
- $n$ is the number of trees.

## Why GBR Is Useful for Hardware Performance Modeling

GPU and LLM performance relationships are often highly nonlinear. For example:

```text
Small GEMM
    -> Often dominated by launch and scheduling overhead

Larger GEMM
    -> May become compute limited

Very large tensor
    -> May expose memory-bandwidth and cache effects

Different data type
    -> May select a completely different kernel
```

A simple linear model such as the following may not capture these transitions effectively:

$$
latency = aM + bN + cK
$$

A tree-based GBR model can instead learn conditional relationships such as:

```text
if datatype == INT2:
    if K > 4096:
        use one performance pattern
    else:
        use another performance pattern
else if datatype == FP16:
    use an FP16-specific pattern
```

Hundreds of small trees can be combined to model complicated interactions among tensor shape, data type, hardware, kernel selection, cache behavior, and runtime overhead.

## Example: Predicting Intel GPU GEMM Performance

Assume we benchmark oneDNN GEMM operations on LNL, PTL, or BMG and collect the following measurements:

| M | N | K | Data type | GPU | Measured latency |
|---:|---:|---:|:---|:---|---:|
| 1 | 4096 | 4096 | INT2 | BMG | 0.35 ms |
| 128 | 4096 | 4096 | INT2 | BMG | 1.80 ms |
| 2048 | 4096 | 4096 | INT2 | BMG | 8.60 ms |
| 1 | 4096 | 4096 | FP16 | BMG | 0.51 ms |

The model training data is organized as:

```text
X = [M, N, K, data_type, GPU, ...]
y = measured_latency
```

After training, the model can estimate latency for a configuration that was not directly benchmarked:

```text
M = 512
N = 4096
K = 4096
data_type = INT2
GPU = BMG
```

The GBR model predicts the expected latency based on patterns learned from previous measurements.

## GBR Versus an Analytical Performance Model

### Roofline Model

```text
FLOPs + memory traffic
        ->
Compute and bandwidth limits
        ->
Estimated performance
```

A Roofline model helps explain **why** a workload is compute-bound or memory-bound.

### GBR Model

```text
Hardware and workload features
        +
Historical benchmark measurements
        ->
Machine-learning model
        ->
Predicted performance
```

A GBR model learns performance behavior empirically. It can capture complex effects without requiring an explicit analytical equation for every kernel and hardware configuration.

## Summary

A GBR performance model is a **data-driven performance predictor**. For Intel GPU and LLM workloads, it can learn how performance changes with:

- Tensor shapes
- Batch size
- Data type and quantization format
- Hardware platform
- Kernel implementation
- Cache and memory-bandwidth behavior
- Runtime and launch overhead

GBR and Roofline models are complementary:

- Use **GBR** to predict performance from benchmark data.
- Use **Roofline analysis** to understand whether performance is limited by computation or memory bandwidth.
