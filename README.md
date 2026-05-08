Unsupervised Neural Network Architectures for Multi-Genre Music Generation

The whole implementation of a neural network-based MIDI music generation is available in this repository. The project is broken up into 4 pieces. Firstly we have started from a basic LSTM Autoencoder and gradually moving toward a Transformer-based music generator and human preference based tuning. This project's primary objective is to create MIDI music samples using various neural network architechture and then compare their performance using plots, evaluation metrics, created outputs, and human input.

We have implemented the project in Google Colab using Python, PyTorch, and MIDI processing libraries.

Task 1: LSTM Autoencoder (Generated short classical piano MIDI samples using reconstruction-based learning)

Task 2: LSTM Variational Autoencoder (Improved diversity using latent sampling and KL-divergence)

Task 3: Transformer Autoregressive Generator (Generated longer MIDI compositions using next-token prediction)

Task 4: RLHF / Human Preference Tuning (Improved generated music using human ratings and reward-guided tuning)

🎵 MIDI Files
All MIDI files used in this project are hosted on Google Drive:
Link: https://drive.google.com/drive/folders/12kGV4w_9QsiFH2VbmTk53dmsJkOTeghq?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto

## Files Included

### Task1_LSTM_Autoencoder.ipynb

This notebook contains the full implementation of Task 1.

It loads classical MIDI files, converts them into piano-roll format, trains an LSTM Autoencoder, reconstructs test samples, and generates 5 new MIDI files. It also includes Random and Markov baseline models for comparison.

Main outputs:

text
task1_outputs/generated_midis/
task1_outputs/plots/
task1_outputs/results/


Important result files:

text
task1_generated_1.mid to task1_generated_5.mid
task1_reconstruction_loss_curve.png
task1_metrics.csv
task1_summary.csv

### Task2_VAE_Music_Generator.ipynb

This notebook contains the full implementation of Task 2.

It implements an LSTM-based Variational Autoencoder. The encoder produces mean and log-variance vectors, and the latent vector is sampled using the reparameterization trick. The model is trained using reconstruction loss and KL-divergence loss.

Main outputs:

text
task2_outputs/generated_midis/
task2_outputs/plots/
task2_outputs/results/


Important result files:

text
task2_vae_generated_1.mid to task2_vae_generated_8.mid
task2_latent_interpolation_1.mid to task2_latent_interpolation_5.mid
task2_vae_total_loss_curve.png
task2_vae_reconstruction_loss_curve.png
task2_vae_kl_loss_curve.png
task2_metrics.csv
task2_summary.csv
task1_vs_task2_metrics_comparison.csv

### Task3_Transformer_Music_Generator.ipynb

This notebook contains the full implementation of Task 3.

For this task, MIDI files are converted into symbolic pitch-duration tokens. A Transformer model is then trained as a next-token predictor using causal attention. The model generates longer MIDI sequences compared to the previous two tasks.

Main outputs:

text
task3_outputs/generated_midis/
task3_outputs/plots/
task3_outputs/results/


Important result files:

text
task3_transformer_generated_1.mid to task3_transformer_generated_10.mid
task3_training_loss_curve.png
task3_perplexity_curve.png
task3_metrics.csv
task3_baseline_comparison.csv
task3_perplexity_report.csv
task3_summary.csv

### Task4_RLHF_Preference_Tuning.ipynb

This notebook contains the full implementation of Task 4.

Task 4 uses human preference feedback to improve the Task 3 Transformer generator. Ten generated MIDI samples were rated by human listeners. The average ratings were used as a reward signal. The model was then tuned using reward-guided preference tuning and a lightweight policy-gradient-style update.

Main outputs:

text
task4_outputs/generated_midis/
task4_outputs/plots/
task4_outputs/results/


Important result files:

text
task4_before_reference_1.mid to task4_before_reference_10.mid
task4_after_tuned_1.mid to task4_after_tuned_10.mid
human_feedback_summary.csv
task4_before_reward_scores.csv
task4_after_reward_scores.csv
task4_policy_gradient_history.csv
task4_before_after_comparison.csv
task4_summary.csv
task4_before_after_reward_score.png


The final reward-based comparison showed improvement after tuning:

text
Before tuning mean reward score: 3.603637
After tuning mean reward score: 3.788676

## Datasets Used

### Task 1 Dataset

Task 1 uses a subset of the *MAESTRO classical piano MIDI dataset*.

The raw dataset is not uploaded to this repository because of file size. During implementation, the MIDI files were stored in Google Drive and loaded into Google Colab.

Expected Drive path:

text
MyDrive/CSE425_Project/task1_midi/classical/

### Task 2 and Task 3 Dataset

Task 2 and Task 3 use a subset of the *Lakh MIDI Clean dataset*. This dataset was used because it contains multi-genre MIDI files.

Expected Drive path:

text
MyDrive/CSE425_Project/task2_midi/multigenre/

### Task 4 Human Feedback

Task 4 uses a human feedback CSV file. The file contains ratings from 10 participants for 10 generated MIDI samples.

Expected format:

text
participant_id,sample_1,sample_2,sample_3,sample_4,sample_5,sample_6,sample_7,sample_8,sample_9,sample_10

Each sample is rated from 1 to 5.

Expected Drive path:

text
MyDrive/CSE425_Project/task4_feedback/human_feedback.csv

## How to Run

This project was developed in *Google Colab*.

### Step 1: Install dependencies

python
%pip -q install pretty_midi mido tqdm

or locally:

bash
pip install -r requirements.txt


### Step 2: Mount Google Drive

python
from google.colab import drive
drive.mount("/content/drive")


### Step 3: Prepare Drive folders

For Task 1:

text
MyDrive/CSE425_Project/task1_midi/classical/

For Task 2 and Task 3:

text
MyDrive/CSE425_Project/task2_midi/multigenre/

For Task 4:

text
MyDrive/CSE425_Project/task4_feedback/human_feedback.csv


### Step 4: Run notebooks in order

Run the notebooks in this order:

text
1. Task1_LSTM_Autoencoder.ipynb
2. Task2_VAE_Music_Generator.ipynb
3. Task3_Transformer_Music_Generator.ipynb
4. Task4_RLHF_Preference_Tuning.ipynb

