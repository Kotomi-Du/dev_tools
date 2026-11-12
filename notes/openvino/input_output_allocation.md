# Defer input allocation

1. compile_model -> network::allocate_primitives() -> BufferA ([PR1](https://github.com/openvinotoolkit/openvino/pull/35126))
2. inferRequest -> allocate_inputs() -> BufferB(ov allocated)  tip: createInferRequest happens in inferenceSession(), right after OpenVINO compilation  ([PR2](https://github.com/openvinotoolkit/openvino/pull/36578), [PR3](https://github.com/openvinotoolkit/openvino/pull/37518))
3. beforeFirstInfer ->BufferC(ovep allocated large buffer) -> setTensor(ovep BufferC+offset) -> BufferB(ov Freed)
4. FirstInfer-> prepare_input() -> setData(ov BufferC)