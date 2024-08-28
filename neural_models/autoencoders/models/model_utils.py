def create_layers(layer_func, nfeat, layer_dims, nout):
    layer_dims.append(nout)
    layers = [layer_func(nfeat, layer_dims[0])]
    for i in range(1, len(layer_dims)):
        layers.append(layer_func(layer_dims[i - 1], layer_dims[i]))
    return layers


def test_create_layers():
    def print_layer(nfeat_in, nfeat_out):
        return "{}x{}".format(nfeat_in, nfeat_out)

    layers = create_layers(print_layer, 10, [], 5)
    assert layers == ['10x5']

    layers = create_layers(print_layer, 10, [2, 3, 4], 5)
    assert layers == ['10x2', '2x3', '3x4', '4x5']
