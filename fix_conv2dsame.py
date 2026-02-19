"""
Post-processing fix for ONNX models exported from timm models with Conv2dSame layers.

torch.export bakes Conv2dSame's dynamic padding into static pads attributes,
which are only correct for the export-time input spatial dimensions.
This script replaces those static pads with dynamic padding subgraphs.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper


def find_conv2dsame_nodes(graph):
    """Find Conv nodes with asymmetric padding (likely from Conv2dSame)."""
    # Build initializer lookup for weight shapes
    initializers = {}
    for init in graph.initializer:
        initializers[init.name] = init

    candidates = []
    for node in graph.node:
        if node.op_type != 'Conv':
            continue
        pads = None
        kernel = None
        stride = None
        for attr in node.attribute:
            if attr.name == 'pads':
                pads = list(attr.ints)
            if attr.name == 'kernel_shape':
                kernel = list(attr.ints)
            if attr.name == 'strides':
                stride = list(attr.ints)

        # If kernel_shape attribute is missing or unreliable,
        # infer from weight tensor shape (Conv weight: [out_ch, in_ch/groups, kH, kW])
        if len(node.input) > 1 and node.input[1] in initializers:
            weight_init = initializers[node.input[1]]
            weight_dims = list(weight_init.dims)
            if len(weight_dims) == 4:
                kernel_from_weight = [weight_dims[2], weight_dims[3]]
                if kernel != kernel_from_weight:
                    kernel = kernel_from_weight

        if pads and kernel and stride:
            if len(pads) == 4:
                # Asymmetric: top != bottom or left != right
                if pads[0] != pads[2] or pads[1] != pads[3]:
                    candidates.append({
                        'node': node,
                        'pads': pads,
                        'kernel': kernel,
                        'stride': stride,
                    })
    return candidates


def make_dynamic_pad_nodes(conv_input, prefix, kernel, stride):
    """
    Create ONNX nodes for dynamic Conv2dSame padding.

    For kernel=k, stride=s:
        out = ceil(input / s)
        pad_total = max(0, (out - 1) * s + k - input)
        pad_begin = pad_total // 2
        pad_end = pad_total - pad_begin

    For the common case k=3, s=2 this simplifies to:
        pad_begin = input_dim % 2   (0 if even, 1 if odd)
        pad_end = 1                 (always)

    For the general case we compute it fully.
    """
    new_nodes = []
    new_initializers = []
    padded_output = f"{prefix}_padded"

    k_h, k_w = kernel
    s_h, s_w = stride

    def make_init(name, arr):
        t = numpy_helper.from_array(arr, name=name)
        new_initializers.append(t)
        return name

    # Scalar/vector constants
    c_0 = make_init(f"{prefix}_c0", np.array([0], dtype=np.int64))
    c_1 = make_init(f"{prefix}_c1", np.array([1], dtype=np.int64))
    c_2 = make_init(f"{prefix}_c2", np.array([2], dtype=np.int64))
    idx2 = make_init(f"{prefix}_idx2", np.array(2, dtype=np.int64))
    idx3 = make_init(f"{prefix}_idx3", np.array(3, dtype=np.int64))
    pad_val = make_init(f"{prefix}_padval", np.array(0.0, dtype=np.float32))

    c_kh = make_init(f"{prefix}_kh", np.array([k_h], dtype=np.int64))
    c_kw = make_init(f"{prefix}_kw", np.array([k_w], dtype=np.int64))
    c_sh = make_init(f"{prefix}_sh", np.array([s_h], dtype=np.int64))
    c_sw = make_init(f"{prefix}_sw", np.array([s_w], dtype=np.int64))

    # Shape(input) -> [N, C, H, W]
    shape_out = f"{prefix}_shape"
    new_nodes.append(helper.make_node('Shape', [conv_input], [shape_out]))

    results = {}
    for dim_label, idx_name, c_k, c_s in [('h', idx2, c_kh, c_sh), ('w', idx3, c_kw, c_sw)]:
        p = f"{prefix}_{dim_label}"

        # Gather dim
        dim_val = f"{p}_dim"
        new_nodes.append(helper.make_node('Gather', [shape_out, idx_name], [dim_val], axis=0))

        dim_1d = f"{p}_1d"
        new_nodes.append(helper.make_node('Reshape', [dim_val, c_1], [dim_1d]))

        # out = ceil(input / stride) = (input + stride - 1) // stride
        add_s_minus_1 = f"{p}_add_sm1"
        s_minus_1 = f"{p}_sm1"
        new_nodes.append(helper.make_node('Sub', [c_s, c_1], [s_minus_1]))
        new_nodes.append(helper.make_node('Add', [dim_1d, s_minus_1], [add_s_minus_1]))

        out_size = f"{p}_out"
        new_nodes.append(helper.make_node('Div', [add_s_minus_1, c_s], [out_size]))

        # pad_total = max(0, (out - 1) * stride + kernel - input)
        out_m1 = f"{p}_outm1"
        new_nodes.append(helper.make_node('Sub', [out_size, c_1], [out_m1]))

        out_m1_x_s = f"{p}_outm1xs"
        new_nodes.append(helper.make_node('Mul', [out_m1, c_s], [out_m1_x_s]))

        plus_k = f"{p}_plusk"
        new_nodes.append(helper.make_node('Add', [out_m1_x_s, c_k], [plus_k]))

        pad_total_raw = f"{p}_ptraw"
        new_nodes.append(helper.make_node('Sub', [plus_k, dim_1d], [pad_total_raw]))

        pad_total = f"{p}_pt"
        new_nodes.append(helper.make_node('Max', [pad_total_raw, c_0], [pad_total]))

        # pad_begin = pad_total // 2
        pad_begin = f"{p}_pb"
        new_nodes.append(helper.make_node('Div', [pad_total, c_2], [pad_begin]))

        # pad_end = pad_total - pad_begin
        pad_end = f"{p}_pe"
        new_nodes.append(helper.make_node('Sub', [pad_total, pad_begin], [pad_end]))

        results[dim_label] = (pad_begin, pad_end)

    # pads = [0, 0, pad_begin_h, pad_begin_w, 0, 0, pad_end_h, pad_end_w]
    pads_name = f"{prefix}_pads"
    new_nodes.append(helper.make_node(
        'Concat',
        [c_0, c_0, results['h'][0], results['w'][0],
         c_0, c_0, results['h'][1], results['w'][1]],
        [pads_name],
        axis=0
    ))

    # Pad
    new_nodes.append(helper.make_node(
        'Pad', [conv_input, pads_name, pad_val], [padded_output], mode='constant'
    ))

    return new_nodes, new_initializers, padded_output


def fix_conv2dsame_padding(onnx_model):
    """
    Replace baked Conv2dSame padding with dynamic padding in-place.
    Returns the number of nodes fixed.
    """
    graph = onnx_model.graph
    candidates = find_conv2dsame_nodes(graph)

    if not candidates:
        return 0

    # Build name -> (index, node) map
    node_list = list(graph.node)
    node_idx = {id(n): i for i, n in enumerate(node_list)}

    insertions = []  # (index, new_nodes, new_inits, padded_name)

    for c in candidates:
        node = c['node']
        idx = node_idx[id(node)]
        prefix = f"dynpad_{idx}"

        new_nodes, new_inits, padded_name = make_dynamic_pad_nodes(
            node.input[0], prefix, c['kernel'], c['stride']
        )
        insertions.append((idx, new_nodes, new_inits, padded_name))

    # Apply in reverse index order
    for idx, new_nodes, new_inits, padded_name in sorted(insertions, key=lambda x: x[0], reverse=True):
        conv_node = node_list[idx]

        # Zero out Conv pads
        for attr in conv_node.attribute:
            if attr.name == 'pads':
                attr.ints[:] = [0, 0, 0, 0]

        # Redirect Conv input
        conv_node.input[0] = padded_name

        # Insert padding nodes before the Conv
        for n in reversed(new_nodes):
            node_list.insert(idx, n)

        # Add initializers
        for init in new_inits:
            graph.initializer.append(init)

    del graph.node[:]
    graph.node.extend(node_list)

    return len(candidates)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fix_conv2dsame.py <input.onnx> [output.onnx]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path

    model = onnx.load(input_path)
    n_fixed = fix_conv2dsame_padding(model)
    print(f"Fixed {n_fixed} Conv2dSame nodes with baked padding")

    onnx.save(model, output_path, save_as_external_data=False)
    print(f"Saved to: {output_path}")

    onnx.checker.check_model(model)
    print("✅ Model check passed")