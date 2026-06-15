# Unsupervised Neural Network for Multi-Genre Music Generation

## Description

This project presents an unsupervised deep learning approach for generating multi-genre music from MIDI data. The system learns musical patterns such as pitch, rhythm, duration, harmony, note progression, and sequence structure without relying on manually labeled data.

The main goal of this project is to generate new MIDI-based music samples across different genres by learning hidden musical representations from existing MIDI sequences.

---

## Motivation

Music is a complex temporal signal that contains structured patterns over time. Traditional supervised learning methods require labeled datasets, which can be expensive and difficult to prepare for music generation tasks.

This project uses unsupervised neural network models to learn music structure directly from MIDI data. By learning latent musical patterns, the models can generate new music samples that preserve rhythm, pitch distribution, and sequential coherence.

---

## Music Sequence Representation

A music sequence is represented as:

$$
X = {x_1, x_2, \dots, x_T}
$$

where each musical event represent:

* Note-on event
* Note-off event
* Pitch
* Velocity
* Duration
* Timing information

The objective is to learn a generative distribution:

$$
p_\theta(X)
$$

so that a new music sequence can be sampled as:

$$
\hat{X} \sim p_\theta(X)
$$

---

## Dataset

This project uses publicly available MIDI datasets for learning musical structure and generating new compositions.

* MAESTRO Dataset

The MIDI files are converted into token-based sequences before training.

---

## Preprocessing

Before training, MIDI files are processed into a model-friendly format.

The preprocessing stage includes:

* MIDI file loading
* Note event extraction
* Pitch, velocity, and duration processing
* Timing normalization
* Piano-roll or token-based representation
* Fixed-length sequence segmentation
* Train-test split preparation

---

## Model 1: LSTM Autoencoder

The LSTM Autoencoder learns a compressed latent representation of a music sequence and reconstructs the original input from that latent representation.

Encoder representation:

$$
z = f_\phi(X)
$$

Decoder reconstruction:

$$
\hat{X} = g_\theta(z)
$$

Reconstruction loss:

$$
L_{AE} = \sum_{t=1}^{T} ||x_t - \hat{x}_t||^2
$$

The Autoencoder is used to learn the basic structure of music sequences and generate short music samples from the latent space.

---

## Model 2: Variational Autoencoder

The Variational Autoencoder learns a probabilistic latent space for generating more diverse music samples.

Latent distribution:

$$
q_\phi(z \mid X) = \mathcal{N}(\mu(X), \sigma^2(X))
$$

Latent sampling:

$$
z = \mu + \sigma \odot \epsilon,\quad \epsilon \sim \mathcal{N}(0, I)
$$

VAE objective:

$$
L_{VAE} = L_{recon} + \beta D_{KL}(q_\phi(z \mid X) ,|, p(z))
$$

The VAE improves generation diversity by sampling from a continuous latent distribution instead of using only fixed encoded features.

---

## Model 3: Transformer-Based Music Generator

The Transformer model generates music autoregressively by predicting the next musical event based on previous events.

Autoregressive probability:

$$
p(X) = \prod_{t=1}^{T} p(x_t \mid x_{1:t-1})
$$

Training loss:

$$
L_{TR} = - \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{1:t-1})
$$

Perplexity:

$$
Perplexity = \exp\left(\frac{1}{T}L_{TR}\right)
$$

The Transformer is used for generating longer and more coherent music sequences by capturing long-range dependencies in MIDI data.

---

## Model 4: Reinforcement Learning with Human Feedback

This stage improves the quality of generated music using human preference scores.

Generated music sample:

$$
X_{gen} \sim p_\theta(X)
$$

Human reward score:

$$
r = HumanScore(X_{gen})
$$

Optimization objective:

$$
J(\theta) = \mathbb{E}[r(X_{gen})]
$$

Policy gradient update:

$$
\nabla_\theta J(\theta) = \mathbb{E}[r \nabla_\theta \log p_\theta(X)]
$$

This model is used to fine-tune the generator so that the generated music better matches human listening preferences.

---

## Evaluation Metrics

The generated music is evaluated using quantitative and qualitative metrics.

### Pitch Histogram Similarity

$$
H(p, q) = \sum_{i=1}^{12} |p_i - q_i|
$$

This metric compares the pitch distribution of real and generated music.

### Rhythm Diversity Score

$$
D_{Rhythm} = \frac{\text{Number of unique durations}}{\text{Total number of notes}}
$$

This metric measures rhythmic variation in the generated music.

### Repetition Ratio

$$
R = \frac{\text{Number of repeated patterns}}{\text{Total number of patterns}}
$$

This metric measures the amount of repeated musical patterns.

### Human Listening Score

$$
Score_{human} \in [1, 5]
$$

This score represents human evaluation of the generated music quality.

---

## Visualization and Analysis

The project includes visual analysis of generated music and model performance.

The visualization part includes:

* Model-average metric comparison
* Rhythm diversity distribution
* Repetition ratio distribution
* Metric correlation heatmap
* Scatter plot for identifying extreme generated files
* Optional pitch similarity aggregation

These visualizations help compare different models and understand the quality of generated MIDI samples.

---

## Outputs

* Generated MIDI music samples
* Model comparison results
* Evaluation metric tables
* Visualization plots
* Human listening score analysis
* Project report

---

## Significance

This project demonstrates how unsupervised neural networks can be used for creative music generation. It explores how different generative models learn musical structure from MIDI data and compares their ability to generate realistic, diverse, and coherent music.

The project also highlights the importance of evaluating generated music using both mathematical metrics and human listening feedback.

---



