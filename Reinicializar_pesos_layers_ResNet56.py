import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
import tensorflow.keras.utils as keras_utils
import tensorflow.keras.backend as backend
from keras.callbacks import LearningRateScheduler
import argparse
import numpy as np
import json
import os
import copy
from datetime import datetime
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics._classification import accuracy_score
from time import time

# TODO depois testar somente reinicializando camadas "principais" dentro de blocos.

def get_allowed_layers(model):
    allowed_layers = []
    all_add = []

    for i in range(0, len(model.layers)):
        if isinstance(model.get_layer(index=i), Add):
            all_add.append(i)

    for i in range(1, len(all_add) - 1):
        input_shape = model.get_layer(index=all_add[i]).output_shape
        output_shape = model.get_layer(index=all_add[i - 1]).output_shape
        # These are the valid blocks we can remove
        if input_shape == output_shape:
            allowed_layers.append(all_add[i])

    # The last block is enabled
    allowed_layers.append(all_add[-1])
    return allowed_layers

class random_score():
    __name__ = 'random_scores'
    def __init__(self):
        pass
    def scores(self,  model, X_train=None, y_train=None, allowed_layers=[]):
        output = []
        for layer_idx in allowed_layers:
            score = np.random.rand()
            output.append((layer_idx, score))
        return output
    
    
class last_layer_score():
    def scores(self, model, X_train=None, y_train=None, allowed_layers=[]):
        output = []
        num_layers = len(allowed_layers)
        for idx, layer_idx in enumerate(allowed_layers):
            score = 1 - (idx / num_layers)
            output.append((layer_idx, score))
        return output
    
class CKA():
    __name__ = 'CKA'
    def __init__(self):
        pass

    def _debiased_dot_product_similarity_helper(self, xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y, n):
        return ( xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y) + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))

    def feature_space_linear_cka(self, features_x, features_y, debiased=False):
        features_x = features_x - np.mean(features_x, 0, keepdims=True)
        features_y = features_y - np.mean(features_y, 0, keepdims=True)

        dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
        normalization_x = np.linalg.norm(features_x.T.dot(features_x))
        normalization_y = np.linalg.norm(features_y.T.dot(features_y))

        if debiased:
            n = features_x.shape[0]
            # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
            sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
            sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
            squared_norm_x = np.sum(sum_squared_rows_x)
            squared_norm_y = np.sum(sum_squared_rows_y)

            dot_product_similarity = self._debiased_dot_product_similarity_helper(
                dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
                squared_norm_x, squared_norm_y, n)
            normalization_x = np.sqrt(self._debiased_dot_product_similarity_helper(
                normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
                squared_norm_x, squared_norm_x, n))
            normalization_y = np.sqrt(self._debiased_dot_product_similarity_helper(
                normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
                squared_norm_y, squared_norm_y, n))

        return dot_product_similarity / (normalization_x * normalization_y)

    def scores(self,  model, X_train=None, y_train=None, allowed_layers=[]):
        output = []

        F = Model(model.input, model.get_layer(index=-2).output)
        features_F = F.predict(X_train, verbose=0)

        F_line = Model(model.input, model.get_layer(index=-2).output)#TODO: Check if this is correct
       #It will probability not work for MobileNetV2 and other convolutional architectures
        for layer_idx in allowed_layers:
            # Resblock: Conv2d, Batch N., Activation, Conv2d, Batch N.
            #if isinstance(model.get_layer(index=layer_idx + self.layer_offset), BatchNormalization):
            #_layer = model.get_layer(index=layer_idx - 1)
            _layer = F_line.get_layer(index=layer_idx - 1)
            _w = _layer.get_weights()
            _w_original = copy.deepcopy(_w)

            print(f"Layer {layer_idx}: {_layer.name} ({_layer.__class__.__name__})")
            
            for i in range(0, len(_w)):
                _w[i] = np.zeros(_w[i].shape)
            _layer.set_weights(_w)
            #F_line = Model(model.input, model.get_layer(index=-2).output)
            features_line = F_line.predict(X_train, verbose=0)

            _layer.set_weights(_w_original)

            score = self.feature_space_linear_cka(features_F, features_line)
            output.append((layer_idx, 1 - score))

        return output

def create_datagen():
    return ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True,
                 name=''):

    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4),
                  name='Conv2D_{}'.format(name))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization(name='BatchNorm1_{}'.format(name))(x)
        if activation is not None:
            x = Activation(activation, name='Act1_{}'.format(name))(x)
    else:
        if batch_normalization:
            x = BatchNormalization(name='BatchNorm2_{}'.format(name))(x)
        if activation is not None:
            x = Activation(activation, name='Act2_{}'.format(name))(x)
        x = conv(x)
    return x

def ResNet(input_shape, depth_block, filters=[],
                 iter=0, num_classes=10):
    num_filters = 16
    i = 0
    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs, num_filters=filters.pop(0))
    i = i + 1
    # Instantiate the stack of residual units
    for stack in range(3):
        num_res_blocks = depth_block[stack]
        for res_block in range(num_res_blocks):
            layer_name = str(stack)+'_'+str(res_block)+'_'+str(iter)
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=filters.pop(0),
                             strides=strides,
                             name=layer_name+'_1')
            i = i + 1
            y = resnet_layer(inputs=y,
                             num_filters=filters.pop(0),
                             activation=None,
                             name=layer_name+'_2')
            i = i + 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=filters.pop(0),
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False,
                                 name=layer_name+'_3')
                i = i + 1
            x = Add()([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def relu6(x):
    return backend.relu(x, max_value=6)

def load_model(architecture_file='', weights_file=''):
    import tensorflow.keras as keras
    from keras.utils.generic_utils import CustomObjectScope
    from keras import backend as K
    from tensorflow.keras import layers

    def _hard_swish(x):
        return x * K.relu(x + 3.0, max_value=6.0) / 6.0

    def _relu6(x):
        return K.relu(x, max_value=6)

    if '.json' not in architecture_file:
        architecture_file = architecture_file+'.json'

    with open(architecture_file, 'r') as f:
        #Not compatible with keras 2.4.x and TF 2.0
        # with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,
        #                         'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D,
        #                        '_hard_swish': _hard_swish}):
        with CustomObjectScope({'relu6': _relu6,
                                'DepthwiseConv2D': layers.DepthwiseConv2D,
                                '_hard_swish': _hard_swish}):
            model = keras.models.model_from_json(f.read())

    if weights_file != '':
        if '.h5' not in weights_file:
            weights_file = weights_file + '.h5'
        model.load_weights(weights_file)
        print('Load architecture [{}]. Load weights [{}]'.format(architecture_file, weights_file), flush=True)
    else:
        print('Load architecture [{}]'.format(architecture_file), flush=True)

    return model

def save_model(file_name='', model=None):
    import keras
    print('Salving architecture and weights in {}'.format(file_name))

    model.save_weights(file_name + '.h5')
    with open(file_name + '.json', 'w') as f:
        f.write(model.to_json())
          
class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if callable(self.model.optimizer.learning_rate):
        # If the learning rate is a callable (schedule), compute it based on the current iteration
            current_lr = self.model.optimizer.learning_rate(self.model.optimizer.iterations)
        else:
            # If the learning rate is a static value, just use it directly
            current_lr = self.model.optimizer.learning_rate

        # Store the learning rate value in logs
        logs['lr'] = current_lr.numpy()
        #logs['lr'] = self.model.optimizer._decayed_lr(tf.float32).numpy()

def scheduler(epoch, lr, warm_up_epochs, target_lr, decay_step, lr_schedule_epochs: list[int]):

    new_lr = target_lr
    if epoch < warm_up_epochs:
        new_lr = target_lr * ((epoch + 1) / warm_up_epochs)
    else:
        for epoch_in_schedule in lr_schedule_epochs:
            if epoch + 1 >= epoch_in_schedule:
                new_lr *= decay_step
    return new_lr



def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print("CIFAR10")
    
    print(len(x_train))
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)

def reinitialize_weights(model, depth_block, filters, scores: list[tuple[int, float]], p_reinitialized_layers):
    input_shape=(32, 32, 3)

    donor_model = ResNet(input_shape, depth_block, filters)
    
        
    indices, score_values = zip(*scores)
    score_values = np.array(score_values)
    sorted_indices = np.argsort(score_values)
    filter_idx_list = sorted_indices[:int(len(score_values) * p_reinitialized_layers)]
    reinitialize_indices = np.array(indices)[filter_idx_list]
    print(f'Indices of reinitialized layers {reinitialize_indices}')
    
    for layer_idx in reinitialize_indices:
        new_weights, new_biases = donor_model.layers[layer_idx-2].get_weights()
        model.layers[layer_idx-2].set_weights([new_weights, new_biases])
        new_weights, new_biases = donor_model.layers[layer_idx-5].get_weights()
        model.layers[layer_idx-5].set_weights([new_weights, new_biases])
    
    
    return model

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--criterion_layer', type=str, default='random')
    parser.add_argument('--p_reinitialized', type=float, default=0)
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--seed_value', type=int, default='')
    parser.add_argument('--timeit', action='store_true', help='times the execution time')

    args = parser.parse_args()
    criterion_layer = args.criterion_layer
    p_reinitialized = args.p_reinitialized
    name = args.name
    seed_value = args.seed_value
    
    if args.timeit:
        start_time = time()
    
    #physical_devices = tf.config.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(physical_devices[0], True) 
    
    
    config = {
        "name": f'{name}_{seed_value}_{criterion_layer}_p_{p_reinitialized}',
        "criterion_layer": criterion_layer,
        "seed_value": seed_value,
        "total_epochs": 150,
        "lr_schedule_epochs": [75, 110],
        "warm_up_epochs": 0,
        "target_lr": 0.01,
        "decay_step": 0.1,
        "p_reinitialized": p_reinitialized
    }
    name                =  config["name"]
    total_epochs        =  config["total_epochs"]
    lr_schedule_epochs  =  config["lr_schedule_epochs"]
    warm_up_epochs      =  config["warm_up_epochs"]
    target_lr           =  config["target_lr"]
    decay_step          =  config["decay_step"]
    p_reinitialized     =  config["p_reinitialized"]

    tf.random.set_seed(seed_value)

    
    ####################################################################
    ####################################################################
    
    (x_train, y_train), (x_test, y_test) = load_data()
    datagen = create_datagen()
   


    model = load_model(f'ResNet56_{seed_value}_50ep.json', f'ResNet56_{seed_value}_50ep')
    
    acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(model.predict(x_test, verbose=0), axis=1))
    print(f"model_acc: {acc}")
 
    
    depth_block = [9, 9, 9]
    filters = [16] * (2*depth_block[0]+1) + [32] * (2*depth_block[1]+1) + [64] * (2*depth_block[2]+1)
    #model = ResNet(input_shape=(32, 32, 3), depth_block=depth_block, filters=filters)


    #lr_logger = LearningRateLogger() # Armazenar lr em history
    lr_callback = LearningRateScheduler(lambda epoch, lr: scheduler(epoch, lr, warm_up_epochs, target_lr, decay_step, lr_schedule_epochs))
    callbacks = [lr_callback] # callbacks = [lr_logger, lr_callback]
    optimizer = SGD(learning_rate=0.01, momentum=0.9, decay=1e-6)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    print(config)
    print(model.summary())
    print(f'{len(x_train) = }')
    print(len(x_train) // 128)
    
    
    for i, layer in enumerate(model.layers):
        print(f"Layer {i}: {layer.name} ({layer.__class__.__name__})", end=", ")
    print("\n")
    
    # Scores
    allowed_layers = get_allowed_layers(model)
    
    if criterion_layer == 'random':
        scores = random_score().scores(model=model, X_train=x_train, allowed_layers=allowed_layers)
    elif criterion_layer == 'CKA':
        scores = CKA().scores(model=model, X_train=x_train, allowed_layers=allowed_layers)
    elif criterion_layer == 'last_layer':
        scores = last_layer_score().scores(model=model, X_train=x_train, allowed_layers=allowed_layers)
    else:
        print("Wrong criterion_layer")
    
    print(scores)

    
    full_data = []

    model = reinitialize_weights(model, depth_block, filters, scores, p_reinitialized)
    
    
    history = model.fit(datagen.flow(x_train, y_train, batch_size=128),
                epochs=total_epochs, callbacks=callbacks,
                verbose=2,
                validation_data=(x_test, y_test))
  
    ## Saving data
    
    history_data = {
            "config": config,
            "history": {k: [float(i) for i in v] for k, v in history.history.items()}
        }
    full_data.append(history_data)


    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("models", exist_ok=True)
    os.makedirs("history", exist_ok=True)
    save_model(f"models/{name}", model)
    filename = f"history/{name}_{current_date}"
    with open(filename, 'w') as f:
        json.dump(full_data, f, indent=4)

    if args.timeit:
        print(f"time(s)  :   {time()- start_time}")