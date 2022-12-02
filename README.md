# CSCI-653: Final Project -- Visualizing Loss Landscape

<p align="center">
  <img src="images/3DExample_rough.jpg" />
  <img src="images/3DExample_smooth.jpg" />
</p>

While training neural nets, the loss function is a function of the model architecture, the optimisation method, initialisation, etc. Yet, the effect of these choices on the resulting objective is unclear. We visualise loss function convergence to gain insights into the training. Visualisation of landscapes offers richer insights and helps explain why neural nets can optimise even extremely complex non-convex functions and why the minimum optimised generalises well.
Given a network architecture and its pre-trained parameters, we calculate and visualize the loss surface along random direction near the optimal parameters. 

## Visualizing 1D loss curve

The 1D linear interpolation method evaluates the loss values along the direction between two minimizers of the same network loss function. This method has been used to compare the flatness of minimizers trained with different batch sizes. A 1D linear interpolation plot would look something like this:

<p align="center">
  <img src="images/1DExample.jpg" />
</p>

## Visualizing 2D loss contours

To plot the loss contours, we choose two random directions with the same dimension as the model parameters and normalize them. A 2D loss contour would look like this:

<p align="center">
  <img src="images/2DExample.jpg" />
</p>

## Things to try

1. Instead of generating random directions from simple Gaussian distribution, use QR decomposition and orthogonalize the directions.

2. Perturb the weights in their first principle directions.

3. There are some cases where even sharp minima can generalize well for deep neural networks [“Sharp Minima Can Generalize For Deep Nets”](https://arxiv.org/pdf/1703.04933.pdf)

4. Parallelize the process of computing loss surface values using MPI (or maybe offload it to GPU)

5. Does this scheme work for a wide range of networks?

## Reference

[1] Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer and Tom Goldstein. [*Visualizing the Loss Landscape of Neural Nets*](https://arxiv.org/abs/1712.09913). NIPS, 2018.

[2] tomgoldstein/loss-landscape. (2020). GitHub. Retrieved December 2, 2022, from [https://github.com/tomgoldstein/loss-landscape](https://github.com/tomgoldstein/loss-landscape)