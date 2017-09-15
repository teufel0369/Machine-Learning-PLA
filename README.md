# Perceptron Learning Algorithm Tutorial

The basis of the Perceptron Learning Algorithm (PLA), and binary classification for that matter, revolves around the idea that if there is no assumption on how past (the training data) is related to the future (the test data), then prediction is impossible.

The relationship in this case is that both past and future observations are both sampled independently from the same distribution.

The prediction boils down to making a weighted estimate (guess), known as the discriminant `D`, where `D = w0 + Σ 𝑤𝑖x𝑖𝑀𝑖=1`.

Consider the feature vectors `X = {x1, x2, x3, ...., xm}` and `Y = {y1, y2, y3, ...., ym}` where `m = the number of dimensions` (or number of elements).

Also consider the weights `W` where `W = {w0, w1, ..., wn}` where `n = the number feature vectors`. Note that `1` bias weight `w0` is always needed.

So the calculation for the discriminant `D` ends up being just a summation of the weights and feature vector elements.

`D = w0 + w1*xm + w2*ym`
`If D >= 0: +1`
`If D < 0: -1`

To be cont'd.
