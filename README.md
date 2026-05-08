Unsupervised Neural Network Architectures for Multi-Genre Music Generation

The whole implementation of a neural network-based MIDI music generation is available in this repository. The project is broken up into 4 pieces. Firstly we have started from a basic LSTM Autoencoder and gradually moving toward a Transformer-based music generator and human preference based tuning. This project's primary objective is to create MIDI music samples using various neural network architechture and then compare their performance using plots, evaluation metrics, created outputs, and human input.

We have implemented the project in Google Colab using Python, PyTorch, and MIDI processing libraries.

Task 1: LSTM Autoencoder (Generated short classical piano MIDI samples using reconstruction-based learning)
Task 2: LSTM Variational Autoencoder (Improved diversity using latent sampling and KL-divergence)
Task 3: Transformer Autoregressive Generator (Generated longer MIDI compositions using next-token prediction)
Task 4: RLHF / Human Preference Tuning (Improved generated music using human ratings and reward-guided tuning)

