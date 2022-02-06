The results include the model name and their corresponding mean absolute error

1. A single feature - single time step
Baseline    : 0.1963,
Linear      : 0.2044,
Dense       : 0.1185,
Multi step dense: 0.1957,
Conv        : 0.1193,
LSTM        : 3.9572

2. Multi-output models - single time step
Baseline       : 0.3100,
Dense          : 0.6923,
LSTM           : 4.1882,
Residual LSTM  : 0.3634

3. Forecasting multiple steps
Baseline: 1.3702,
Repeat baseline: 2.0283,
Single shot linear: 1.5519,
Dense: 2.7096,
CNN: 2.4752,
RNN: 6.4491,
Autoregressive model: 6.8681
