"""Learned galaxy morphology: AutoEncoder + Normalizing Flow prior.

Ported from the ``pshear`` package (Train-AE repository). Provides a
frozen, pretrained decoder that maps latent codes to galaxy images, and a
frozen normalizing flow that serves as the NumPyro-compatible prior over
those latent codes. Used as a drop-in replacement for the parametric
Sersic/Exponential renderer on the smallest stamp tier (see
``shine.euclid.scene``).
"""
