# Simple-Neural-Network

YouTube Hampus Liked Videos
https://www.youtube.com/playlist?list=PLAyUwmL7et7M7tPcEyCtxfXweb0Dwd7g4

YouTube Sentex
https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3

Stochastic, Mini-Batch and Batch Gradient Descent
https://www.youtube.com/watch?v=S-xOow1e2hg&t=19s

Forward Propagation:

Calculate the Weighted sum of the Inputs + Biases and feed the
result to an Activation Function for each Node in each Layer

The Input Values of each Layer is the Output Values of the Layer before

The Weights is Stored as a Matrix for each Layer. The Matrix has
as many Rows (Height) as Output Nodes, and as many Columns (Width)
as Input Nodes.

The Biases is Stored as a Vector (Array) with one float for every Output Node

You should use Different Activation Functions for Hidden Layers and the Output Layer

Examples of Activation Functions for Hidden Layers:
- ReLU
- Sigmoid
- Tanh
- Step

Examples of Activation Functions for Output Layers:
- For Regression: linear (because values are unbounded)
- For Classification: Softmax (simple sigmoid works too)

Use simple sigmoid only if your output admits multiple "true"
answers, for instance, a network that checks for the presence of
various objects in an image. In other words, the output is not a
probability distribution (does not need to sum to 1).

To calculate the output of a normal Activation Function, you
only need to input one value. But to calculate the output of
Softmax, you need to input every input node at once, and
calculate every output at once. This means that you must
calculate the derivate of each input all at once through a
special Softmax derivate function.

https://stats.stackexchange.com/questions/218542/which-activation-function-for-output-layer

https://www.youtube.com/watch?v=sNTtUV9yE_M

Back Propagation Stochastic Gradient Descent:

Really Good Explanation of Back Propagation
https://www.youtube.com/watch?v=kbGu60QBx2o&list=PLAyUwmL7et7M7tPcEyCtxfXweb0Dwd7g4&index=10&t=585s

Calculate the Cost (also called Error or Loss) using one of many Cost Functions:
- For Regression Problem:
- For Binary Classification: Binary Cross Entropy
- For Multi-Class Classification: Cross Categorical Entropy

https://www.youtube.com/watch?v=NJpABYQB9PI&list=PLAyUwmL7et7M7tPcEyCtxfXweb0Dwd7g4&index=1

Calculate the Partial Derivative of the Cost Function with respect
to Each Parameter (Weights and Biases) using the Chain Rule:

https://www.youtube.com/watch?v=hfMk-kjRv4c&list=PLAyUwmL7et7M7tPcEyCtxfXweb0Dwd7g4&index=6&t=1250s

https://www.youtube.com/watch?v=-zI1bldB8to&list=PLAyUwmL7et7M7tPcEyCtxfXweb0Dwd7g4&index=2&t=514s

One Weight can have Influence on the Cost Function Through
Different Paths. We calculate Partial Derivatives for the
different Paths and add them together.

https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmiro.medium.com%2Fmax%2F1183%2F1*bLcTD-AxVnQfoO8qlZ-4sQ.jpeg&f=1&nofb=1&ipt=8b1e8b566c34153e4d1643f3980f43a0a73d25b41190f723b07c7cfa3e086af9&ipo=images

Super Good Explanation of Back Propagation and Gradient calculation:
https://www.youtube.com/watch?v=sIX_9n-1UbM

1. Feed Forward (calculate all sums and activations for the layer nodes and store them) calculate outputs
- store sum nodes as the biases (matrix) [is not important if the activation function is normal]
- store activation nodes values as the biases (matrix)
2. Calculate the error (targets - outputs)
3. Calculate derivatives for every weight from last to first layer (dL/dwi)
4. Calculate deltas using derivatives, learnRate, momentum and last deltas
5. Update weight and biases with deltas

Update Weights and Biases using Gradient Descent:

Walking down the Cost Function in the direction of the
Negative Partial Derivatives, by taking small steps

Update the Parameters (Weights and Biases), by subtract the
Negative Partial Derivatives times a small Learning Rate and
adding the Last Change (Delta Weight) times Momentum.

D Wi(t) = -n(dE/dWi) + a * Dwi(t-1)

The Minus sign -n(dE/dWi) can Instead be added in the calculation of the Gradient

https://www.youtube.com/watch?v=IruMm7mPDdM

https://stats.stackexchange.com/questions/70101/neural-networks-weight-change-momentum-and-weight-decay
