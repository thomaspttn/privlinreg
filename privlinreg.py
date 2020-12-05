import tensorflow as tf
import pandas as pd
import numpy as np
from scipy import stats


class PrivLinReg:
    '''
    CLASS - PrivLinReg: Creates a model for differentially private
    linear regression
    '''

    def __init__(self, df, epsilon, noise_type):
        self.df = df
        self.noise_type = noise_type
        self.epsilon = epsilon

        self.x = self.df.iloc[:, :-1]
        self.y = self.df.iloc[:, -1].astype('float32')
        self.dataset = tf.data.Dataset.from_tensor_slices((self.x.to_numpy(),
                                                           self.y.to_numpy()))

        self.delta_f = (2 * (df.shape[1] + 1)**2) / self.epsilon

        if self.noise_type == 'laplace':
            lap = stats.laplace(scale=self.delta_f)
            self.noise_vals = abs(lap.rvs(size=3))
        elif self.noise_type == 'gauss':
            sp = ((2 * np.log(1.25))**0.5 * self.delta_f) / self.epsilon
            gauss = stats.norm(scale=sp)
            self.noise_vals = abs(gauss.rvs(size=3))
        else:
            self.noise_vals = [0, 0, 0]

        # print(self.noise_vals)

    def train_model(self, epochs=10, learning_rate=0.001):
        '''
        Create a linear regression model and train it on the
        class's training data.

        :param int epochs: How many iterations to train the model for
        :param float learning_rate: How quickly to adjust the model
        :return model: The tf.Keras trained model
        '''
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(1, activation='linear',
                                  dtype='float32', use_bias=False,
                                  )
        ])

        adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss=self.noisy_objective,
                      optimizer=adam,
                      metrics=['accuracy'])

        model.fit(self.x,
                  self.y,
                  batch_size=64,
                  epochs=epochs,
                  verbose=0)

        return model

    def noisy_objective(self, y_true, y_pred):
        '''
        Add appropriate noise to the objective function for optimization.
        Note: Return abs() to ensure loss is positive. Else objective
        will fail.

        :param int y_true: The true class of the prediction
        :param float y_pred: The predicted class of the prediction
        :return: The objective function evaluated for this prediciton
        '''
        y2_term = (y_true**2) + self.noise_vals[0]
        xy_term = (y_true * y_pred) + self.noise_vals[1]
        x2_term = (y_pred**2) + self.noise_vals[2]
        return abs(y2_term - (2 * xy_term) + x2_term)
