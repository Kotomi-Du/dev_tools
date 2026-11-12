# WeightlessCacheAttribute in GPU and NPU Weightless Caching

This document describes how `ov::WeightlessCacheAttribute` (WCA) connects an
`ov::op::v0::Constant` to its original weight data during weightless compiled-model
caching.

WCA contains three fields:

```cpp
size_t original_size;
size_t bin_offset;
ov::element::Type original_dtype;
```

Their meaning depends on the model source:

| Model source | `bin_offset` meaning | Weight source during import |
| --- | --- | --- |
| OpenVINO IR | Real byte offset in the IR `.bin` file | Original `ov::Model` or mapped `.bin` |
| ONNX/non-IR | Synthetic Constant identifier | Original converted `ov::Model` |

WCA is metadata only. It does not own weight data and is not itself the compiled
weightless blob format. GPU and NPU consume WCA while compiling and encode the
required identity into their own compiled representations.

The attribute definition and explicit copy helper are in
[`weightless_caching_attributes.hpp`](src/core/dev_api/openvino/core/rt_info/weightless_caching_attributes.hpp)
and
[`weightless_caching_attributes.cpp`](src/core/src/op/util/weightless_caching_attributes.cpp).

## 1. IR Model

An OpenVINO IR normally consists of an XML graph and a `.bin` weights file. A
Constant's XML `<data>` entry contains its byte offset, size, and element type.

### Three-Stage Workflow

| Plugin | Compile | Export | Import |
| --- | --- | --- | --- |
| GPU | Core creates WCA with real `.bin` offsets; GPU transfers valid WCA into each `cldnn::data` cache manager. | Each weightless `data` primitive stores its offset and replay metadata instead of Constant bytes. | GPU resolves the offset through the original model or mapped `.bin`, then replays conversion/reorder into GPU memory. |
| NPU | NPU requires WCA, persists it as `ws_*` model RT fields, and compiles main and weight-initialization schedules. | NPU writes the main schedule followed by init schedules; original weight buffers are omitted. | NPU maps init-input IDs to model Constants or mapped `.bin` ranges, runs the init schedules, and connects their outputs to the main graph. |

### 1.1 WCA Creation

The IR deserializer reads the Constant metadata and attaches:

```cpp
ov::WeightlessCacheAttribute(original_size, bin_offset, original_dtype)
```

to the Constant's runtime information. See
[`XmlDeserializer::parse_weightless_cache_attribute()`](src/core/xml_util/src/xml_deserialize_util.cpp)
and the attachment of the result to `ovNode` in the same file.

For IR, `bin_offset` is a real byte offset in the `.bin` file. Two Constants may
legitimately share an offset when they share the same underlying weight bytes.

### 1.2 GPU

#### Compile

1. Core reads the XML and `.bin`, creating Constants with WCA.
2. GPU clones the model and runs transformations.
3. Transformations preserve WCA only when the replacement Constant still represents
   the same source bytes.
4. `ProgramBuilder::add_primitive()` reads WCA from each Constant-backed
   `cldnn::data` primitive and calls:

   ```cpp
   data_prim->cache_info->set_constant_info(
       attr.bin_offset,
       attr.original_size,
       attr.original_dtype,
       op.get_output_element_type(0),
       op.get_output_shape(0));
   ```

See
[`ProgramBuilder::add_primitive()`](src/plugins/intel_gpu/src/plugin/program_builder.cpp)
and
[`weightless_cache_manager`](src/plugins/intel_gpu/include/intel_gpu/primitives/data.hpp).

If a Constant no longer has WCA after transformation, its `data` primitive takes the
ordinary cache path and embeds its bytes in the compiled blob.

#### Export

`cldnn::data::save()` invokes `weightless_cache_manager::save()`. For a weightless
Constant it serializes:

- a weightless marker;
- the original `.bin` offset;
- precision-conversion metadata when the compiled Constant type differs;
- reorder metadata when GPU preprocessed the Constant layout.

The Constant bytes are omitted. If weightless metadata is unavailable,
`cldnn::data::save()` writes the bytes normally.

This logic is in
[`data::save()`](src/plugins/intel_gpu/include/intel_gpu/primitives/data.hpp).

#### Import

`program::load()` creates a `WeightsMemory` source. It supports two forms:

1. **Model-backed:** if the original `ov::Model` is available, `WeightsMemory`
   builds `WCA.bin_offset -> ov::Constant`.
2. **File-backed:** otherwise, if `ov::weights_path` is available, GPU memory-maps
   the `.bin` file and interprets `bin_offset` as a byte offset.

See [`program::load()`](src/plugins/intel_gpu/src/graph/program.cpp) and
[`WeightsMemory`](src/plugins/intel_gpu/include/intel_gpu/primitives/data.hpp).

For every weightless `cldnn::data` primitive, `load_weights()` reads the serialized
offset and obtains either:

```text
mapped_bin.data + bin_offset, original_size
```

or:

```text
offset_to_constant_map[bin_offset]->get_data_ptr()
```

It then reproduces any recorded precision conversion and GPU reorder before filling
the destination memory.

### 1.3 NPU

#### Compile

NPU does not synthesize missing WCA in its standard weights-separation flow. Before
calling `compileWS()`, it checks that at least one Constant carries WCA and throws if
none is present. See
[`check_weightless_cache_attribute_occurrence()`](src/plugins/intel_npu/src/plugin/src/plugin.cpp).

The NPU serializer runs its compatibility transformations first and then converts
each WCA into deterministic model runtime fields:

```text
ws_bin_offset_<constant-id>
ws_original_size_<constant-id>
ws_original_dtype_<constant-id>
```

See
[`storeWeightlessCacheAttribute()`](src/plugins/intel_npu/src/compiler_adapter/src/model_serializer.cpp).

The NPU compiler uses that information to produce:

- a main inference schedule;
- one or more weight initialization schedules;
- initialization input descriptors whose `nameFromCompiler` identifies the source
  weight by its WCA offset.

#### Export

`WeightlessGraph::export_blob()` writes the main compiled schedule followed by the
weight initialization schedules. The large original weight buffers are not included.

The exported blob does not need to contain live WCA objects. The association derived
from WCA is represented by the compiled initialization-schedule metadata.

See
[`WeightlessGraph::export_blob()`](src/plugins/intel_npu/src/compiler_adapter/src/weightless_graph.cpp).

#### Import

The presence of initialization schedules tells NPU that the imported blob uses
weights separation. NPU accepts either the original model or an `ov::weights_path`.

For an original `ov::Model`,
`get_all_constants_in_topological_order()` builds:

```text
WCA.bin_offset -> ov::Constant
```

For a `.bin` path, `get_all_constants_memory_mapped()` parses each initialization
descriptor's `nameFromCompiler` as the original byte offset and creates Constants
which view the corresponding mapped `.bin` range.

Both paths are selected by `extract_constants_map()` in
[`weightless_graph.cpp`](src/plugins/intel_npu/src/compiler_adapter/src/weightless_graph.cpp).

`WeightlessGraph::allocate_inputs()` then:

1. parses the expected weight ID from `descriptor.nameFromCompiler`;
2. finds that ID in `_constants`;
3. validates the byte size;
4. copies the Constant bytes into an initialization-schedule input;
5. runs the initialization schedule;
6. connects its processed outputs to weight inputs of the main schedule.

Thus NPU may not inspect WCA objects during `.bin` import, but WCA was required during
compilation to encode the same offsets into the initialization schedules.

## 2. Non-IR Model

In this document, non-IR primarily means an ONNX or ONNX Runtime model converted into
an `ov::Model`. GPU and NPU do not execute directly from the raw ONNX protobuf. The
ONNX frontend first produces `ov::Constant` nodes.

### Three-Stage Workflow

| Plugin | Compile | Export | Import |
| --- | --- | --- | --- |
|current GPU | GPU uses frontend-provided WCA(bin_offset = topology-ID), or set WCA (bin_offset = incremental index for constant only) when no Constant has WCA, related code: **ov::WeightlessCacheAttribute(1, offset++, node->get_element_type())**; | Each weightless `data` primitive stores the synthetic ID and replay metadata instead of Constant bytes. | GPU uses the converted original `ov::Model` and maps the synthetic ID back to an `ov::Constant`, related code: **offset_to_constant_map.emplace(attr.bin_offset, const_ptr)**; Constants without WCA are read from bytes embedded in the blob. |
| ONNX FE change | no WCA for internal constants if there is external weights file | |  |
| GPU adapt to ONNX FE change |skip setting WCA for internal constants if there is external weights file;the other remains same behavior with before| |no change needed, because the map update is based on if there is WCA  |
| NPU | NPU requires frontend-provided WCA, persists the synthetic IDs as `ws_*` RT fields, and compiles main and weight-init schedules. | NPU writes the main and init schedules without the original Constant buffers. | NPU uses the supplied converted model, or rereads an ONNX path, builds `WCA.bin_offset -> ov::Constant`, and feeds matching Constants to init schedules. |

For inline and external ONNX initializers, these plugin workflows are identical. The
ONNX frontend resolves the original storage before either plugin receives the
`ov::Model`.

### 2.1 Synthetic WCA

The ONNX frontend's `enumerate_constants()` visits `model->get_ordered_ops()` and
assigns every Constant:

```cpp
ov::WeightlessCacheAttribute(0, operation_index, constant_element_type)
```

See
[`enumerate_constants()`](src/frontends/onnx/frontend/src/frontend.cpp).

The synthetic `bin_offset` is therefore the Constant's topological index among all
operations in the converted OpenVINO graph. It is not:

- an ONNX initializer index;
- an ONNX node index;
- an external-data file offset;
- a memory address;
- the ordinal among Constants only.

The conversion performed for cache creation and cache import must produce compatible
WCA identifiers and Constant values.

### 2.2 Inline and External ONNX Constants

The plugins do not distinguish an inline ONNX initializer from an ONNX external-data
initializer.

The ONNX frontend owns that distinction. Depending on the frontend mode, it may:

- wrap memory supplied by ONNX Runtime;
- memory-map an ONNX external-data file;
- read external data into frontend-owned memory;
- materialize inline protobuf data.

All forms become `ov::Constant` storage before GPU or NPU receives the model. The
plugins retrieve bytes through `ov::Constant::get_data_ptr()`.

See
[`extract_tensor_external_data()`](src/frontends/onnx/frontend/src/core/graph_iterator_proto.cpp),
[`TensorExternalData`](src/frontends/onnx/frontend/src/utils/tensor_external_data.cpp),
and [`tensor_has_data()`](src/frontends/onnx/frontend/src/translate_session.cpp).

Consequently, ONNX external-data offsets are not directly encoded into synthetic WCA.
The converted `ov::Model` remains responsible for keeping the Constant storage alive.

### 2.3 GPU

#### Compile

If the frontend already supplied WCA, GPU uses it. This is the normal ONNX frontend
path.

For another non-IR integration that supplies no WCA, GPU has a temporary fallback:
`set_weightless_cache_attributes()` assigns sequential synthetic offsets to Constants
on the cloned model. Unlike ONNX frontend enumeration, this fallback counter advances
only for Constants.

See
[`Plugin::set_weightless_cache_attributes()`](src/plugins/intel_gpu/src/plugin/plugin.cpp).

GPU transformations and `ProgramBuilder` then follow the same path as for IR:

```text
synthetic WCA -> data::cache_info
```

#### Export

The GPU blob stores the synthetic offset and any required conversion/reorder metadata
inside each `cldnn::data` primitive. It omits that Constant's bytes. It does not
serialize an `ov::Model` containing WCA.

#### Import

Core passes the converted original `ov::Model` through `ov::hint::model`. GPU uses
one of two mappings:

1. If Constants already carry WCA, `WeightsMemory` directly builds
   `WCA.bin_offset -> ov::Constant`.
2. If GPU had to synthesize WCA itself, import creates a temporary
   `GpuWeightlessCacheMap` of `Constant::instance_id -> synthetic WCA`, stores it in
   the internal `GPU_WEIGHTLESS_ATTR` configuration property, and uses it to build
   the same `synthetic offset -> ov::Constant` lookup.

See
[`Plugin::create_weightless_cache_attributes()`](src/plugins/intel_gpu/src/plugin/plugin.cpp),
[`GpuWeightlessCacheMap`](src/plugins/intel_gpu/include/intel_gpu/runtime/internal_properties.hpp),
and
[`WeightsMemory::fill_offset_to_constant_map()`](src/plugins/intel_gpu/include/intel_gpu/primitives/data.hpp).

The serialized synthetic offset selects the original Constant, and GPU copies or
reprocesses `Constant::get_data_ptr()` into GPU memory.

### 2.4 NPU

#### Compile

NPU requires WCA to have already been supplied by the frontend. It does not have the
GPU fallback that stamps a model with new synthetic attributes.

For ONNX, `enumerate_constants()` provides topological operation indices. NPU verifies
WCA presence, persists its fields as `ws_*` model RT information, and invokes
`compileWS()` to create the main and initialization schedules.

#### Export

As with IR, NPU exports the main schedule and initialization schedules. The synthetic
WCA IDs become initialization input identities in compiler metadata; the original
ONNX Constant bytes are omitted.

#### Import

NPU can receive either:

- the converted original `ov::Model`; or
- an `.onnx` `ov::weights_path`.

For a model, NPU builds `WCA.bin_offset -> ov::Constant` directly. For an `.onnx`
path, NPU calls `core->read_model()` first; the ONNX frontend recreates Constants and
synthetic WCA, after which NPU builds the same map.

The compiler-generated initialization descriptor name is parsed as the synthetic ID
and used to select the Constant in `WeightlessGraph::allocate_inputs()`.

Inline and external ONNX Constants take the same NPU path after frontend conversion.
NPU does not reopen or interpret an ONNX external-data file itself when an `ov::Model`
is supplied. When an `.onnx` path is supplied, Core and the ONNX frontend resolve its
external data while recreating the `ov::Model`.

## 3. WCA During GPU Transformations

`WeightlessCacheAttribute::is_copyable()` returns `false`. Generic
`ov::copy_runtime_info()` therefore does not automatically transfer WCA to replacement
nodes.

This is intentional. A transformation may fold, convert, transpose, reorder, or
otherwise change Constant bytes. Blindly retaining the original offset would cause
cache import to restore bytes that no longer represent the transformed Constant.

### 3.1 Explicit Preservation

When a transformation knows that a replacement Constant is only a shape view or can
be reconstructed from the same original data, it explicitly calls:

```cpp
ov::copy_weightless_cache_attr(source, replacement);
```

The helper supports:

1. direct copying when `source` carries WCA;
2. copying from the Constant input when `source` is a `Convert` folded into a new
   Constant.

Important call sites include:

- core Constant Folding in
  [`constant_folding.cpp`](src/core/src/pass/constant_folding.cpp);
- precision conversion in
  [`convert_precision.cpp`](src/common/transformations/src/transformations/convert_precision.cpp);
- compressed 1x1 Convolution-to-MatMul conversion in
  [`convert_weight_compressed_conv1x1_to_matmul.cpp`](src/common/transformations/src/transformations/op_conversions/convert_weight_compressed_conv1x1_to_matmul.cpp);
- GPU compressed FullyConnected conversion in
  [`convert_fc_to_compressed.cpp`](src/plugins/intel_gpu/src/plugin/transformations/convert_fc_to_compressed.cpp).

For example, constructing a Constant with a new shape from the same Constant storage
does not alter the byte sequence:

```cpp
auto new_constant = std::make_shared<ov::op::v0::Constant>(*constant, new_shape);
ov::copy_weightless_cache_attr(constant, new_constant);
```

Preserving WCA here allows GPU to omit the reshaped Constant from the cache and restore
the original bytes later. The current shape is separately recorded in
`weightless_cache_manager` when needed for reconstruction.

### 3.2 Loss of WCA Is a Safe Fallback

For GPU, the decision is made independently for each Constant-backed `cldnn::data`
primitive. If a pass preserves WCA for some Constants but not others, the resulting
mixed cache remains correct:

```text
Constant A has WCA    -> export stores its source offset
Constant B has no WCA -> export embeds its transformed bytes
Constant C has WCA    -> export stores its source offset
```

For a Constant without WCA, the fallback is:

```text
no WCA
  -> ProgramBuilder does not enable weightless caching for that data primitive
  -> export embeds the transformed Constant bytes
  -> import reads those bytes from the compiled blob
```

Missing WCA therefore reduces the cache-size benefit for that Constant, but avoids
restoring stale or semantically incorrect bytes.

For non-IR models, there is one additional nuance. GPU's
`Plugin::is_weightless_cache_attributes_set()` returns `true` when any Constant has
WCA. GPU then does not synthesize WCA for the remaining Constants. Those Constants
still safely use the embedded-data fallback, so the result can be a partially
weightless blob. See
[`Plugin::is_weightless_cache_attributes_set()`](src/plugins/intel_gpu/src/plugin/plugin.cpp).

Missing WCA is safe, but an incorrect or stale WCA is not. If a transformation changes
Constant bytes and retains the old source identity, export may omit the changed bytes
and import may restore the wrong original data.

This per-Constant fallback is explicit in the GPU implementation. NPU only verifies
that at least one Constant has WCA before entering weights separation; support for a
partially attributed model additionally depends on the NPU compiler's generated init
schedule contract and should not be assumed from the plugin-side check alone.

### 3.3 MoE Offload Extension

GPU MoE offload also needs original `.bin` offsets after transformations. Before the
pipeline, GPU stamps a plain `otd_bin_offset` RT value on relevant Constants. Plain RT
values survive generic `copy_runtime_info()`, unlike WCA.

The MoE path resolves an offset in this order:

1. direct WCA;
2. Constant source-buffer descriptor;
3. `otd_bin_offset` fallback.

See
[`TransformationsPipeline::apply()`](src/plugins/intel_gpu/src/plugin/transformations_pipeline.cpp)
and [`CreateMoEOp()`](src/plugins/intel_gpu/src/plugin/ops/moe.cpp).

This MoE fallback is adjacent to, but distinct from, the normal GPU compiled-model
weightless cache flow.

## Q&A About the New Proposal

### What Is the Proposal?

- ONNX FE sets WCA bin_offset = (external) ? valid values : ORT model topological index
- OVEP removes WCA from internal Constants that are not stored in the external file.

### Why Separate External and Internal Constants?

The old design uses a Constant's topological index as `bin_offset`. If two models use
the same external weights file, the same weight can have different indices:

```text
Model A: shared weight W -> bin_offset 12
Model B: shared weight W -> bin_offset 7
```

These values cannot locate W in the shared file and do not show that both models use
the same bytes. With the proposed design, both models store W's real file offset:

```text
Model A: shared weight W -> bin_offset 4096
Model B: shared weight W -> bin_offset 4096
                         -> shared_weights.bin + 4096
```

Internal Constants have no corresponding bytes in `shared_weights.bin`, so OVEP
removes their WCA and the plugin embeds them in each compiled blob. Thus only external
weights are restored from the shared file.

### Confusion
* when ONNX FE update to valid value? compile ?  who report this?
* involve OVEP removal or not?



