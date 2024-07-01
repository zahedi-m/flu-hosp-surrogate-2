# flu-hosp-surrogate
flu hospitalization surrogate model with Gaussian uncertainty quantification
The reconstruction loss is negative log likelihood of the Gaussian distribution
The encoder uses latent prevalence
there are four decoders for each output compartments
The decoder uses GRU, and the hidden state of the GRU serves as the autoregressive structure