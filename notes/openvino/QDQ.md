
## Simple explanation

`fq->input_value(0)` means the output of the first input connected to the
`FakeQuantize` node. Usually, this is the original data flowing into FQ.

`fq->output(0)` means the output produced by the `FakeQuantize` node.

When calling:

```cpp
output.replace(replacement);
```

the consumer receives data from replacement, intead of output ; it does not immediately delete the old node.

For example:

```text
data ──> FakeQuantize ──> Relu ──> Result
                 │
                 └── output(0)
```
Assume
```
output      = fq->output(0);
replacement = fq->input_value(0);  // data->output(0)

Before:
Relu.input(0) ← FakeQuantize.output(0)

After:
Relu.input(0) ← data.output(0)
```

Now it becomes:

```text
data ───────────────> Relu ──> Result
```

## FQ replacement topology rejected by `has_result_consumers(replacement)`

The direct replacement is rejected when the `FakeQuantize` output is a model
output and the node feeding the FQ is already used by another model output:

```text
                    ┌──> Result_A
data ──> InputOp ────┤
                    └──> FakeQuantize ──> Result_B
```

For this call:

```cpp
replace_output_update_name(fq->output(0), fq->input_value(0));
```

`fq->output(0)` has a `Result` consumer (`Result_B`), and
`fq->input_value(0)` is `InputOp->output(0)`, which already has a `Result`
consumer (`Result_A`). Therefore:

```cpp
has_result_consumers(replacement) == true
```

The replacement is refused because directly reconnecting `Result_B` to
`InputOp` would make the existing replacement output feed multiple model
results and could change or conflict with friendly names:

```text
                    ┌──> Result_A
data ──> InputOp ────┤
                    └──> Result_B   # rejected direct replacement
```

The fallback creates a fresh `Identity` node:

```cpp
auto identity = std::make_shared<ov::op::v16::Identity>(
    fq->input_value(0));
```

This gives `Result_B` a separate output path:

```text
                    ┌──> Result_A
data ──> InputOp ────┴──> Identity ──> Result_B
```

The new `Identity` has no existing `Result` consumers, so it can safely
replace the `FakeQuantize` output while preserving the FQ friendly name.

## FQ replacement topology rejected for a `Parameter`

The direct replacement is also rejected when the FQ input is a
`Parameter` and the FQ output is directly connected to a model result:

```text
Parameter ──> FakeQuantize ──> Result
```

In this call:

```cpp
replace_output_update_name(fq->output(0), fq->input_value(0));
```

the replacement is `Parameter->output(0)`. Since `fq->output(0)` has a
`Result` consumer, `can_output_be_replaced()` checks whether the replacement
node is a `Parameter` and rejects it:

```cpp
ov::is_type<ov::op::v0::Parameter>(replacement.get_node()) == true
```

The direct replacement would be:

```text
Parameter ───────────> Result   # rejected direct replacement
```

Instead, the fallback inserts an `Identity`:

```text
Parameter ──> Identity ──> Result
```

The `Identity` is a single-output, non-`Parameter` replacement with no
existing `Result` consumers. It allows the FQ output to be safely rewired
while preserving the original FQ friendly name on the model-output path.

## Possible risks of replacing an FQ output with a `Parameter`

Consider this graph:

```text
Parameter "input_data" ──> FakeQuantize "output_quantized" ──> Result
```

If the FQ output were replaced directly with the `Parameter` output, the
graph would become:

```text
Parameter "input_data" ──> Result
```

When the output has a `Result` consumer, the replacement node may receive the
old FQ friendly name. Renaming the `Parameter` from `"input_data"` to
`"output_quantized"` could create these risks:

* Application code that looks up the input by `"input_data"` may fail.
* Input and output names may become confusing or collide.
* A transformation intended to remove an internal FQ operation could
  unintentionally change the model's public input/output interface.
* Model inspection, serialization, debugging, profiling, or tests that depend
  on friendly names may change.

The `Identity` fallback avoids this by keeping the input name unchanged and
giving the old FQ output name to a new intermediate node:

```text
Parameter "input_data" ──> Identity "output_quantized" ──> Result
```