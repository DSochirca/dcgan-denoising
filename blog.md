# DCGANs for image super-resolution, denoising and deblurring
Dan Sochirca, Petter Reijalt

In this blog post, we emulate the findings in the paper DCGANs for image super-resolution, denoising and deblurring [^1]. We sought to improve on the suggested architecture by introducing and leveraging self-attention. 

## Introduction
Increased computational power and the advent of bigger datasets have improved the ability of deep learning methods to do image processing. Generative adversarial networks (GANs) have been created to learn deep representations without needing extensively annotated training data. GANs can learn by implicitly figuring out the similarity between the distribution of a certain candidate model and the distribution of the real data. A so-called _generator_ tries to 'fool' the _discriminator_ by emulating a sample from the real data set. The _discriminator_ tries to differentiate between samples produced by the _generator_ and drawn from the real data distribution. Early GANs made use of fully connected neural networks for both the generator and the discriminator. This naturally evolved into convolutional GANs, as Convolutional Neural Networks (CCNs) are well suited to image data [^2]. This resulted in Deep Convolutional GANs (DCGANs), which makes use of strided and fractionally strided convolutions. These DCGANs have been used to great success in building up good image representations [^3]. << link to denosing and gap filling >>

## Background
- GANS
- DCGANS
- Denoising

## Methodology

## Results

## Conclusion


[^1]: [DCGANs for image super-resolution, denoising and debluring](https://stanford.edu/class/ee367/Winter2017/yan_wang_ee367_win17_report.pdf)
[^2]: [Generative Adversarial Networks: An Overview](https://ieeexplore.ieee.org/abstract/document/8253599)
[^3]: [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
