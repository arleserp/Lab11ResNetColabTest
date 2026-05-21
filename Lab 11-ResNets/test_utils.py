from termcolor import colored

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Dense


# Compare the two inputs
def comparator(learner, instructor):
    if len(learner) != len(instructor):
        raise AssertionError(
            f'Models does not have the same number of layers {len(learner)} != {len(instructor)}'
        )

    for a, b in zip(learner, instructor):
        if tuple(a) != tuple(b):
            print(
                colored("Test failed", attrs=['bold']),
                "\n Expected value \n\n",
                colored(f"{b}", "green"),
                "\n\n does not match the input value: \n\n",
                colored(f"{a}", "red")
            )
            raise AssertionError("Error in test")

    print(colored("All tests passed!", "green"))


# Extracts the description of a given model
def summary(model):

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    result = []

    for layer in model.layers:

        layer_name = layer.__class__.__name__

        # Manejo compatible con TensorFlow moderno
        if hasattr(layer, 'output') and hasattr(layer.output, 'shape'):

            # TensorShape -> tuple
            shape = tuple(layer.output.shape)

            # Compatibilidad con notebooks antiguos
            if layer_name == "InputLayer":
                output_shape = [shape]
            else:
                output_shape = shape

        else:
            output_shape = 'N/A'

        # Parámetros
        try:
            param_count = layer.count_params()
        except:
            param_count = 'N/A'

        descriptors = [layer_name, output_shape, param_count]

        # Conv2D
        if type(layer) == Conv2D:
            descriptors.append(layer.padding)
            descriptors.append(layer.activation.__name__)
            descriptors.append(layer.kernel_initializer.__class__.__name__)

        # MaxPooling2D
        if type(layer) == MaxPooling2D:
            descriptors.append(layer.pool_size)
            descriptors.append(layer.strides)
            descriptors.append(layer.padding)

        # Dropout
        if type(layer) == Dropout:
            descriptors.append(layer.rate)

        # ZeroPadding2D
        if type(layer) == ZeroPadding2D:
            descriptors.append(layer.padding)

        # Dense
        if type(layer) == Dense:
            descriptors.append(layer.activation.__name__)

        result.append(descriptors)

    return result