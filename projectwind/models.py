import tensorflow as tf

class Baseline_model(tf.keras.Model):
    """
    Baseline model is the average of the input timesteps (repeated over width of the target).
    This is done by calculating a simple mean over the input historical target.
    ------------
    Returns:
    Predictions of the baseline model
    """
    def __init__(self, window):
        super().__init__()
        self.n_steps_out = window.label_width
        self.label_indices = window.label_columns_indices[window.label_columns[0]]

    def call(self, inputs):
        inputs = inputs[:, :, self.label_indices]
        input_mean = tf.math.reduce_mean(inputs)
        prediction = tf.repeat(input_mean,self.n_steps_out)
        prediction = prediction[None, :, tf.newaxis]
        return prediction


class LSTM_Regressor_Model(tf.keras.Model):
    def __init__(self, units, n_steps_out, num_features):
        super().__init__()
        # Variables
        self.n_steps_out = n_steps_out
        self.units = units
        
        # Layers
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        self.dense = tf.keras.layers.Dense(num_features)
          
    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the LSTM state.
        prediction, state = self.warmup(inputs)
        # Insert the first prediction.
        predictions.append(prediction)

        # Run the rest of the prediction steps.
        for n in range(1, self.n_steps_out):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state,
                                      training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output.
            predictions.append(prediction)

            # predictions.shape => (time, batch, features)
            predictions = tf.stack(predictions)
            # predictions.shape => (batch, time, features)
            predictions = tf.transpose(predictions, [1, 0, 2])
        
        prediction = tf.keras.layers.Dense(self.n_steps_out)(predictions)
        
        return predictions

    def compile_and_fit(name, model, window, patience=5, epoch=100):

        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        patience=patience,
                                                        mode='min',
                                                        restore_best_weights=True)

        # Reduce learning rate by an order of magnitude if val_loss does not improve for 20 epoch
        rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                    factor=0.1,
                                                    min_lr=1e-7,
                                                    verbose=1,
                                                    patience=10)

        # Model checkpoint
        checkpoint=tf.keras.callbacks.ModelCheckpoint(f"./checkpoint/Feedback_Model_{model_name}.h5", 
                                                    save_best_only=True,
                                                    save_weights_only=True)

        model.compile(loss=tf.losses.MeanSquaredError(),
                    optimizer=tf.optimizers.Adam(),
                    metrics=[tf.metrics.MeanAbsoluteError()])

        history = model.fit(window.load_train, epochs=epoch,
                            validation_data=window.load_val,
                            callbacks=[early_stopping, rlrop, checkpoint])
        return history


class LSTM_Classifier_Model(tf.keras.Model):
    def __init__(self, units, n_steps_out, num_features):
        super().__init__()
        # Variables
        self.n_steps_out = n_steps_out
        self.units = units
        
        # Layers
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        self.dense = tf.keras.layers.Dense(num_features)
          
    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the LSTM state.
        prediction, state = self.warmup(inputs)
        # Insert the first prediction.
        predictions.append(prediction)

        # Run the rest of the prediction steps.
        for n in range(1, self.n_steps_out):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state,
                                      training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output.
            predictions.append(prediction)

            # predictions.shape => (time, batch, features)
            predictions = tf.stack(predictions)
            # predictions.shape => (batch, time, features)
            predictions = tf.transpose(predictions, [1, 0, 2])
        
        prediction = tf.keras.layers.Dense(self.n_steps_out)(predictions)
        
        return predictions

    def compile_and_fit(name, model, window, patience=5, epoch=100):

        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        patience=patience,
                                                        mode='min',
                                                        restore_best_weights=True)

        # Reduce learning rate by an order of magnitude if val_loss does not improve for 20 epoch
        rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                    factor=0.1,
                                                    min_lr=1e-7,
                                                    verbose=1,
                                                    patience=10)

        # Model checkpoint
        checkpoint=tf.keras.callbacks.ModelCheckpoint(f"./checkpoint/Feedback_Model_{model_name}.h5", 
                                                    save_best_only=True,
                                                    save_weights_only=True)

        model.compile(loss=tf.losses.MeanSquaredError(),
                    optimizer=tf.optimizers.Adam(),
                    metrics=[tf.metrics.MeanAbsoluteError()])

        history = model.fit(window.load_train, epochs=epoch,
                            validation_data=window.load_val,
                            callbacks=[early_stopping, rlrop, checkpoint])
        return history


class Feedback_Model(tf.keras.Model):
    """
    Feedback / auto-regressive model.
    This class allows the construction of an LSTM auto-regressive model (i.e. decomposes model's output into individual steps
    to use the predicted output of timestep n, as an input to predict timestep n + 1).  
    ------------
    Parameters:
    units: 'int' 
           Number of LSTM (auto-regressive) layers of the model

    ------------
    Returns:
    Prediction for timeseries of n_steps_out
    """
    def __init__(self, units, n_steps_out, num_features):
        super().__init__()
        # Variables
        self.n_steps_out = n_steps_out
        self.units = units
        
        # Layers
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        self.dense = tf.keras.layers.Dense(num_features)
        
    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)

        # Wrap the LSTMCell in an RNN to simplify the `warmup` method.
        lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True) 
        
        # x.shape => (batch, lstm_units)
        x, *state = lstm_rnn(inputs)
        
        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        
        return prediction, state
    
    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the LSTM state.
        prediction, state = self.warmup(inputs)
        # Insert the first prediction.
        predictions.append(prediction)

        # Run the rest of the prediction steps.
        for n in range(1, self.n_steps_out):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state,
                                      training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output.
            predictions.append(prediction)

            # predictions.shape => (time, batch, features)
            predictions = tf.stack(predictions)
            # predictions.shape => (batch, time, features)
            predictions = tf.transpose(predictions, [1, 0, 2])
        
        prediction = tf.keras.layers.Dense(self.n_steps_out)(predictions)
        
        return predictions

    def compile_and_fit(name, model, window, patience=5, epoch=100):

        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        patience=patience,
                                                        mode='min',
                                                        restore_best_weights=True)

        # Reduce learning rate by an order of magnitude if val_loss does not improve for 20 epoch
        rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                    factor=0.1,
                                                    min_lr=1e-7,
                                                    verbose=1,
                                                    patience=10)

        # Model checkpoint
        checkpoint=tf.keras.callbacks.ModelCheckpoint(f"./checkpoint/Feedback_Model_{model_name}.h5", 
                                                    save_best_only=True,
                                                    save_weights_only=True)

        model.compile(loss=tf.losses.MeanSquaredError(),
                    optimizer=tf.optimizers.Adam(),
                    metrics=[tf.metrics.MeanAbsoluteError()])

        history = model.fit(window.load_train, epochs=epoch,
                            validation_data=window.load_val,
                            callbacks=[early_stopping, rlrop, checkpoint])
        return history



