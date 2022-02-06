This notebook builds a few different styles of models including Convolutional and Recurrent Neural Networks (CNNs and RNNs).

This is covered in two main parts, with subsections: 

* Forecast for a single time step:
  * A single feature.
  * All features.
* Forecast multiple steps:
  * Single-shot: Make the predictions all at once.
  * Autoregressive: Make one prediction at a time and feed the output back to the model.

The results include the model name and their corresponding mean absolute error

1. A single feature - single time step

Baseline    : 0.1963
Linear      : 0.2044
Dense       : 0.1185
Multi step dense: 0.1957
Conv        : 0.1193
LSTM        : 3.9572

2. Multi-output models - single time step
Baseline       : 0.3100
Dense          : 0.6923
LSTM           : 4.1882
Residual LSTM  : 0.3634


3. Forecasting multiple steps
Baseline: 1.3702
Repeat baseline: 2.0283
single shot linear: 1.5519
dense: 2.7096
cnn : 2.4752
rnn: 6.4491
Autoregressive model: 6.8681

