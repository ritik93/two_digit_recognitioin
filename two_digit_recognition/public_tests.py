# UNIT TESTS
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Dense

import numpy as np

def test_c1(target):
    assert len(target.layers) == 3, \
        f"Wrong number of layers. Expected 3 but got {len(target.layers)}"
    assert target.input.shape.as_list() == [None, 400], \
        f"Wrong input shape. Expected [None,  400] but got {target.input.shape.as_list()}"
    i = 0
    expected = [[Dense, [None, 25], sigmoid],
                [Dense, [None, 15], sigmoid],
                [Dense, [None, 1], sigmoid]]

    for layer in target.layers:
        assert type(layer) == expected[i][0], \
            f"Wrong type in layer {i}. Expected {expected[i][0]} but got {type(layer)}"
        assert layer.output.shape.as_list() == expected[i][1], \
            f"Wrong number of units in layer {i}. Expected {expected[i][1]} but got {layer.output.shape.as_list()}"
        assert layer.activation == expected[i][2], \
            f"Wrong activation in layer {i}. Expected {expected[i][2]} but got {layer.activation}"
        i = i + 1

    print("\033[92mAll tests passed!") 
