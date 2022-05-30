import tensorflow as tf

class Baseline_model(tf.keras.Model):
    """
    Baseline model is the mean of the input historical target, repeated over width of the target window.
    ------------
    Returns:
    Predictions of the baseline model
    """
    def __init__(self, window):
        super().__init__()
        self.n_steps_out = window.label_width
        self.label_indices = window.label_columns_indices[window.label_columns[0]]

    def call(self, inputs):
        # Select historical target column values, but keep shape (batch, time, feature)
        x = tf.expand_dims(inputs[:, :, self.label_indices], axis=-1)
        
        # Calculate the mean of historical target 
        x_mean = tf.math.reduce_mean(x, axis=1, keepdims=True)
        
        # Repeat mean across output timesteps
        predictions = tf.tile(x_mean[:,:,:], [1, self.n_steps_out, 1])
        return predictions


def lstm_regressor_model(window):
    # Performance model
    input_perf = tf.keras.layers.Input(shape=window.train['X'].shape[1:])
    x_seq = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(input_perf)
    x_seq = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(x_seq)
    x_perf = tf.keras.layers.LSTM(64, return_sequences=False, dropout=0.2)(x_seq)
    
    # Weather forecast model
    input_fc = tf.keras.layers.Input(shape=window.train['X_fc'].shape[1:])
    x_fc_seq = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(input_fc)
    x_fc_seq = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(x_fc_seq)
    x_fc = tf.keras.layers.LSTM(64, return_sequences=False, dropout=0.2)(x_fc_seq)
    
    # Combined model
    combined = tf.keras.layers.concatenate([x_perf, x_fc], axis=1)
    x = tf.keras.layers.Dense(128, activation='relu')(combined)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(window.train['y'].shape[1], activation="relu")(x)
    
    # Reshape (batch, sequences, features)
    outputs = tf.keras.layers.Reshape(window.train['y'].shape[1:])(outputs)
    
    model = tf.keras.models.Model(inputs=[input_perf, input_fc], outputs=outputs)
    
    return model


def lstm_classifier_model(window):
    # Performance model
    input_perf = tf.keras.layers.Input(shape=window.train['X'].shape[1:])
    x_seq = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(input_perf)
    x_seq = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(x_seq)
    x_perf = tf.keras.layers.LSTM(64, return_sequences=False, dropout=0.2)(x_seq)
    
    # Weather forecast model
    input_fc = tf.keras.layers.Input(shape=window.train['X_fc'].shape[1:])
    x_fc_seq = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(input_fc)
    x_fc_seq = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(x_fc_seq)
    x_fc = tf.keras.layers.LSTM(64, return_sequences=False, dropout=0.2)(x_fc_seq)
    
    # Combined model
    combined = tf.keras.layers.concatenate([x_perf, x_fc], axis=1)
    x = tf.keras.layers.Dense(128, activation='relu')(combined)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
    
    # Reshape (batch, sequences, features)
    outputs = tf.keras.layers.Reshape(window.train['y'].shape[1:])(outputs)
    
    model = tf.keras.models.Model(inputs=[input_perf, input_fc], outputs=outputs)
    
    return model

def gru_regressor_model(window):
    # Performance model
    input_perf = tf.keras.layers.Input(shape=window.train['X'].shape[1:])
    x_seq = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(input_perf)
    x_seq = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(x_seq)
    x_perf = tf.keras.layers.LSTM(64, return_sequences=False, dropout=0.2)(x_seq)
    
    # Weather forecast model
    input_fc = tf.keras.layers.Input(shape=window.train['X_fc'].shape[1:])
    x_fc_seq = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(input_fc)
    x_fc_seq = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(x_fc_seq)
    x_fc = tf.keras.layers.LSTM(64, return_sequences=False, dropout=0.2)(x_fc_seq)
    
    # Combined model
    combined = tf.keras.layers.concatenate([x_perf, x_fc], axis=1)
    x = tf.keras.layers.Dense(128, activation='relu')(combined)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(window.train['y'].shape[1], activation="relu")(x)
    
    # Reshape (batch, sequences, features)
    outputs = tf.keras.layers.Reshape(window.train['y'].shape[1:])(outputs)
    
    model = tf.keras.models.Model(inputs=[input_perf, input_fc], outputs=outputs)
    
    return model


def gru_classifier_model(window):
    # Performance model
    input_perf = tf.keras.layers.Input(shape=window.train['X'].shape[1:])
    x_seq = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(input_perf)
    x_seq = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(x_seq)
    x_perf = tf.keras.layers.LSTM(64, return_sequences=False, dropout=0.2)(x_seq)
    
    # Weather forecast model
    input_fc = tf.keras.layers.Input(shape=window.train['X_fc'].shape[1:])
    x_fc_seq = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(input_fc)
    x_fc_seq = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(x_fc_seq)
    x_fc = tf.keras.layers.LSTM(64, return_sequences=False, dropout=0.2)(x_fc_seq)
    
    # Combined model
    combined = tf.keras.layers.concatenate([x_perf, x_fc], axis=1)
    x = tf.keras.layers.Dense(128, activation='relu')(combined)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
    
    # Reshape (batch, sequences, features)
    outputs = tf.keras.layers.Reshape(window.train['y'].shape[1:])(outputs)
    
    model = tf.keras.models.Model(inputs=[input_perf, input_fc], outputs=outputs)
    
    return model





class LSTM_Model(tf.keras.Model):
  def __init__(self, window, units=32, hidden_layers=3):
    super(LSTM_Model, self).__init__()
    # Variables
    self.window = window
    self.units = units
    self.hidden_layers = hidden_layers
   
    # Models
    self.performance_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=self.window.train['X'].shape[1:]),
        tf.keras.layers.LSTM(units, return_sequences=True, dropout=0.2),
        tf.keras.layers.LSTM(units, return_sequences=False, dropout=0.2)
    ])
    self.forecast_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=self.window.train['X_fc'].shape[1:]),
        tf.keras.layers.LSTM(units, return_sequences=True, dropout=0.2),
        tf.keras.layers.LSTM(units, return_sequences=False, dropout=0.2)
    ])
    self.combined_model = tf.keras.Sequential([
        tf.keras.layers.Concatenate(),
        tf.keras.layers.Dense(units*2, activation='relu'),
        tf.keras.layers.Dense(units, activation='relu'),
        tf.keras.layers.Dense(self.window.train['y'].shape[1]),
        tf.keras.layers.Reshape(self.window.train['y'].shape[1:])
    ])

  def call(self, inputs):
    
    # Run train['X'] through performance model
    print(inputs[0].shape)
    perf_predictions = self.performance_model(inputs[0])

    # Run train['X_fc'] through forecast model
    print(inputs[1].shape)
    fc_predictions = self.forecast_model(inputs[1])

    # Run combined output into final model
    predictions = self.combined_model([perf_predictions, fc_predictions])

    return predictions




#  TODO: Complete the LSTM model builders
class LSTM_Model_3(tf.keras.Model):
    def __init__(self, window, shape, units=32, hidden_layers=3):
        super().__init__()
        # Variables
        self.window = window
        self.units = units
        self.hidden_layers = hidden_layers
        self.shape = shape
        print(self.shape)
        print(self.shape[1:])

        # Define layers
        self.input_layer = tf.keras.layers.Input(shape=self.shape[1:])
        self.lstm_hidden_layer = tf.keras.layers.LSTM(units, return_sequences=True, dropout=0.2)
        self.lstm_outer_layer = tf.keras.layers.LSTM(units, return_sequences=False, dropout=0.2)
        self.dense_1 = tf.keras.layers.Dense(units*2, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(units, activation='relu')
        self.dense_final = tf.keras.layers.Dense(self.window.train['y'].shape[1])
        self.reshape_layer = tf.keras.layers.Reshape(self.window.train['y'].shape[1:])

    def call(self, input, training=False):
        
        print('model_input:', input)
        print('model_input_shape:', input.shape)
        
        input_x = self.input_layer(input)
        print(input_x)
        # for i in self.hidden_layers:
        #     print(i)
        x = self.lstm_hidden_layer(input_x)
        prediction = self.lstm_outer_layer(x)
        
        print('model_output:',prediction)
        print('model_output_shape:', prediction.shape)
        
        return prediction




class LSTM_Model_Builder(tf.keras.Model):
    def __init__(self, window, units=32, hidden_layers=3):
        super(LSTM_Model_Builder, self).__init__()
        # Variables
        self.window = window
        self.units = units
        self.hidden_layers = hidden_layers

        # Define layers 
        self.input_layer = tf.keras.layers.Input(shape=self.window.train['X'].shape[1:])
        self.input_layer_fc = tf.keras.layers.Input(shape=self.window.train['X_fc'].shape[1:])
        self.lstm_inner_layer = tf.keras.layers.LSTM(units, return_sequences=True, dropout=0.2)
        self.lstm_outer_layer = tf.keras.layers.LSTM(units, return_sequences=False, dropout=0.2)
        self.dense_1 = tf.keras.layers.Dense(units*2, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(units, activation='relu')
        self.dense_final = tf.keras.layers.Dense(self.window.train['y'].shape[1])
        self.reshape_layer = tf.keras.layers.Reshape(self.window.train['y'].shape[1:])

        # Define models
        self.perf_model = Performance_Model(self.window, self.units, self.hidden_layers)

    # def build(self, window):
    #     input_perf_shape = window.train['X'].shape[1:]
    #     input_fc_shape = window.train['X_fc'].shape[1:]

    #     input_perf = tf.keras.layers.Input(shape=input_perf_shape)
    #     input_fc = tf.keras.layers.Input(shape=input_fc_shape)

    #     output_perf = self.perf_model(input_perf)
    #     output_fc = self.forecast_model(input_fc)

    #     combined_output = tf.keras.layers.concatenate([output_perf, output_fc], axis=1)

    #     predictions = self.combined_model(combined_output)

    #     return predictions

    class Performance_Model(tf.keras.Model):
        def __init__(self, window, units=32, hidden_layers=3):
            super(Performance_Model, self).__init__()
            # Variables
            self.window = window
            self.units = units
            self.hidden_layers = hidden_layers
            # Performance model
            input_perf = inputs[0]
            print('perf_model:', input_perf)
            x = self.input_layer_perf(input_perf)
            x = self.lstm_inner_layer(x)
            x = self.lstm_inner_layer(x)
            output = self.lstm_outer_layer(x)
            print('perf_model_output:',output)
            return output
    
    def forecast_model(self, inputs, training=False):
        # Weather forecast model
        input_fc = inputs[1]
        print('fc_model:',input)
        x = self.input_layer_fc(input_fc)
        x = self.lstm_inner_layer(x)
        x = self.lstm_inner_layer(x)
        output = self.lstm_outer_layer(x)
        print('fc_model_output:',output)
        return output
        
    def combined_model(self, input, training=False):
        # Combined model
        x = self.dense_1(input)
        x = self.dense_2(x)
        outputs = self.dense_final(x)
        
        # Reshape (batch, sequences, features)
        predictions = self.reshape_layer(outputs)
                
        return predictions
                
    def call(self, inputs, training=False):

        input_perf = inputs[0]
        input_fc = inputs[1]
        print(input_perf)
        print(input_fc)

        x_perf = self.input_layer_perf(input_perf)
        x_perf = self.lstm_inner_layer(x_perf)
        x_perf = self.lstm_inner_layer(x_perf)
        y_perf = self.lstm_outer_layer(x_perf)
        print('perf_model_output:',y_perf)

        combined_outputs = tf.keras.layers.concatenate([y_perf, output_fc], axis=1)

        predictions = self.combined_model(combined_outputs, training)

        return predictions



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






