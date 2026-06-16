Machine learning
competition for market
state forecasting
Predict the next market state
from a sequence of prior states.
Solve the problem quants face daily.
SEP 15 — DEC 1
PRIZE POOL $13,600
Finals
This challenge is over. Congratulations to the winners!

1
insuperabilehart
0.3964
$5,000
2
stiv_yakovenko
0.3958
$2,500
3
cteceliker
0.3951
$1,700
4
Artem_Voronov
0.3942
$1,300
5
sultanmunirov
0.3932
$1,000
6
pavlylko
0.3924
$800
7
denisalpino
0.39214
$700
8
Mehul
0.39212
$600
Separate congratulations to all the other active participants, the competition was really tight. We hope to see you again next time!

UPDATE
The next time is here: Wunder Predictorium

The task
In this competition, you are invited to build a model that predicts the next market state from a sequence of prior states.

This is a very challenging endeavor due to the complexity of the data: many standard time-series statistical assumptions are not met here. Yet the problem is feasible — Wunder Fund's 10 years of successful trading prove it.

The task mirrors problems quantitative researchers face daily. You'll have to be smart — in the HFT domain, inference needs to be made under very tight time constraints, so any practical solution must be nimble enough to run on CPU.

Timeline and prizes
TIMELINE
START
SEP 15
SUBMISSIONS CLOSED
DEC 1
FINISH
DEC 15
The competition will take 2.5 months.
— SEP 15, 2025: competition starts
— DEC 1, 2025: submissions are no longer accepted
In the period from DEC 1, 2025 to DEC 15, 2025 all the solutions will be rescored on the validation dataset the winners will be determined and rewarded with prizes.
PRIZES
1
$5,000
2
$2,500
3
$1,700
4
$1,300
5
$1,000
6
$800
7
$700
8
$600
Current leaders
#	name	best score	subs	last sub
1	insuperabilehart	0.3920	447	2025-12-01
2	stiv_yakovenko	0.3910	76	2025-12-01
3	cteceliker	0.3897	62	2025-12-01
4	aayaann	0.3880	247	2025-12-01
5	sultanmunirov	0.3876	162	2025-12-01
6	Mehul	0.3872	163	2025-12-01
7	denisalpino	0.3871	193	2025-12-01
8	Goshawinprize	0.3870	308	2025-12-01
9	Artem_Voronov	0.3870	101	2025-12-01
10	misha	0.3869	50	2025-11-29
Full leaderboard →
The data
The dataset consists of precomputed anonymized features that closely resemble those we use in production across some of the markets we trade.

Each market state is represented by a vector of N=32 features. You will receive independent sequences of states; for each sequence, use the first 100 market states to infer the current regime, then predict the next state at each step using the full history of that sequence.

All sequences come from a single market but span different time periods and market conditions. The training set includes 517 sequences, each with 1,000 states.

HOSTED BY
Wunder Fund Logo
WUNDER FUND
Wunder Fund is a global high frequency trading firm. We operate since 2014 and we run our strategies on many traditional and crypto markets.

High frequency trading is highly competitive and genuinely fun, and it really suits people with hacker mindsets who like fast feedback loops.

Welcome and have fun.
— WF team. Quick start
Welcome to the Wunder Challenge! This guide will walk you through the first steps, from setting up your environment to making your first submission.

1. Get the starter pack
Use this shell oneliner:

macOS/Linux
Windows

curl -o wnn_starterpack.zip https://files.wundernn.io/wnn_starterpack.zip && \
tar -xf wnn_starterpack.zip && \
cd competition_package 
or download the archive manually:

STARTERPACK
wnn_starterpack.zip
wnn_starterpack.tar.gz

slow download? try mirror-1/mirror-2
What's inside
Inside the archive you'll find competition_package folder:


competition_package/
├── datasets
├   └── train.parquet
├── examples
├   └── simple
├       ├── README.md
├       └── solution.py
├── README.md
└── utils.py
Here's some files worth noting:

datasets/train.parquet: a training dataset to build and test your models locally.
utils.py: contains helper classes (like DataPoint) and the scoring function.
examples/simple/solution.py: a minimal working example to show the required submission format.
2. Set up your environment
We strongly recommend using a virtual environment to keep your project dependencies tidy and avoid conflicts.

macOS/Linux
Windows

python -m venv env 
source env/bin/activate 
Now, any packages you install will be contained in this private environment.

3. Install dependencies
To work with the data and run the baseline solution, you'll need a few libraries. Install them using pip:


pip install pandas scikit-learn pyarrow numpy tqdm
4. Run the simple example
The competition package includes a simple example to help you understand the basic data processing loop.

Go to the example folder:

cd competition_package
cd examples/simple
Run the script:

python solution.py
The script will read the sample training data, generate predictions, and print the R² score to your screen. It's a great starting point for understanding the data format and the task.

You may notice a few things about the example solution that are critical. Your further submissions will need to meet these requirements as well:

NOTE
The main file must be solution.py.
It must contain a class named PredictionModel with a predict method.
You can include other files in your solution too, like model weights, config files, or helper modules.
Make sure solution.py is at the root level of your submisson.
5. Prepare and submit solution
When your solution is ready, you need to package it as a .zip archive. The solution.py file must be at the root of the archive.

macOS/Linux
Windows
Navigate to your solution's folder and run this command:


zip -r submission.zip .
Go to the submit page and send your solution for scoring.

🎉 ta-daa, you're awesome
Now build your own solution
The simple example is just a starting point. To compete for the top places, you’ll need to train your own model.

TIP
Transformer models or recurrent architectures like LSTM, GRU, and Mamba-2 are well-suited for this kind of sequence modeling task.
 
TIP
Create a validation set
To test your model’s performance before submitting, you’ll need a validation set. Because all sequences in the data are independent and shuffled, you can easily create one. Just split the sequences by their seq_ix. For example, use 80% of the sequences for training and 20% for validation.
Good luck with the challenge! We're excited to see what you build. Data overview
Understanding the data is key to success in this challenge. Here’s a detailed breakdown of the dataset format, structure, and key properties.

To get the data you need to sign up and go to quick start page.
Data format
The entire dataset is provided as a single table in a Parquet file. Each row represents a single market state at a specific point in time.

The table has N + 3 columns:

seq_ix: The ID of the sequence. This is an integer that identifies which sequence the row belongs to.
step_in_seq: An integer representing the step number within a sequence, from 0 to 999.
need_prediction: A boolean (True or False). If True, you need to provide a prediction for the next step.
N feature columns: The remaining N columns are the anonymized numeric features that describe the market state.
The sequences
The data is organized into many independent sequences.

Sequence length: Each sequence is exactly 1000 steps long (from step_in_seq 0 to 999). Prizes and verification
We have a total prize pool of over $13,000 USDT for the top 8 participants on the Private Leaderboard.

Prize pool
The prizes will be distributed as follows:

+--------+-------------+
| Place  | Prize, USDT |
+--------+-------------+
| 1st    |      $5,000 |
| 2nd    |      $2,500 |
| 3rd    |      $1,700 |
| 4th    |      $1,300 |
| 5th    |      $1,000 |
| 6th    |        $800 |
| 7th    |        $700 |
| 8th    |        $600 |
+--------+-------------+
| Total  |     $13,600 |
+--------+-------------+

Prize eligibility and verification
To be eligible for a prize, the top 8 participants must pass a solution verification process. After the preliminary results are announced, you will have 7 calendar days to submit your materials.

What you'll need to provide
Inference code: The exact code that produced your winning submission, including any model weights.
Training code: The complete code used to train your model, with fixed random seeds.
Technical report: A short (2–8 page) document describing your approach, architecture, and validation strategy.
Code review call: A 30-60 minute video call with us to walk through your code.
License: You'll grant us a non-exclusive license to use your winning solution internally for research purposes. Your work will not be made public.
How we'll verify your solution
We will check for:

Reproducibility: We can run your code and get the same results.
Compliance: Your solution follows all competition rules.
Clarity: Your technical report clearly explains your method.
Prize payout
Once your solution is verified, we'll transfer the prize in USDT to your wallet within 14 days.

If verification fails, we'll ask for corrections. If the issues can't be resolved within 5 days, the prize will be offered to the next-highest-ranking participant. Competition rules
Welcome to the Wunder Challenge! We're excited to see what you'll build. Here are the rules to ensure a fair and fun competition for everyone.

The basics
Who can join: The competition is open to everyone, worldwide. You can participate as an individual or as part of a team.
How you're scored: We use two leaderboards. The Public Leaderboard provides feedback on a sample of the test data. The Private Leaderboard, used at the end, determines the final winners on a separate, hidden dataset.
What you submit: This is a code competition. You'll upload your inference code as a .zip file.
Fair play
No external data: Your solution must only use the training data we provide. Using other datasets, pre-trained models, or external embeddings is not allowed.
Isolated environment: Your code will run in an isolated Linux container with no internet access.
Sharing: Please don't share your code, models, or detailed strategies publicly before the competition is over. Discussing general ideas is great, but sharing complete solutions isn't fair to others. Private sharing between participants is also not allowed.
Submissions: You can make up to 10 submissions per day.
Single accounts: One account per team/person. We won't see the #1 participant to take all the winning ranks.
No messing with the infra: We respect out-of-the box thinking and the hacker way very much. If you crack our scoring system, leak the data or such — you'll have our regards, but you'll also be disqualified.
Violation of the rules will lead to disqualification.
Technical rules
Execution environment: Your code will run in a Linux container with Python 3.10+. A detailed list of installed libraries will be provided.
Resources: The environment is CPU-only.
Time limit: Your solution must complete its predictions on the test set in 60 minutes or less.
Reproducibility: Your code must be deterministic. Running it twice on the same data should produce the same results. Remember to fix your random seeds.
Leaderboard and tie-breaking
Final ranking: Your final position will be determined solely by your score on the Private Leaderboard.
Tie-breaking: If two participants have the exact same score, the participant who submitted their solution earlier will be ranked higher. FAQ
Here are answers to some common questions. This list will be updated during the competition.

Competition mechanics
What is the goal of this competition?
Your goal is to predict the next market state vector from a sequence of previous states. It's a sequence modeling task.

What's the evaluation metric?
We use the R² (coefficient of determination) score, averaged across all N features in the state vector. A higher score is better.

Can I work in a team?
Yes, you can participate as an individual or as part of a team.

How many submissions can I make per day?
You can make up to 10 submissions per day.

Data questions
Can I use external data or pre-trained models?
No. All solutions must be trained using only the provided train.parquet dataset. Using external data or pre-trained models found online is not allowed.

Why are the features anonymized?
The features are anonymized to focus the competition on the modeling task itself, rather than on domain-specific feature engineering that would be difficult to anonymize.

How should I create a validation set?
The sequences in the data are independent and shuffled. This means you can create a reliable validation set by splitting the data based on seq_ix. For example, use 80% of sequences for training and 20% for validation.

Technical questions
What are the compute resources for the submission environment?
Your code will run in a Linux container with:

1 CPU core
16 GB of RAM
No GPU
A 60-minute time limit for the entire test set.
Why is there no GPU?
In many real-world, high-frequency trading environments, inference must happen very quickly on CPU-only hardware. We've designed the competition environment to reflect these realistic constraints.

What Python libraries are available?
We will provide a requirements.txt file detailing the exact Python environment. You can expect standard libraries like numpy, pandas, scikit-learn, torch, and tensorflow.

Does my code need to be deterministic?
Yes. Your code should produce the same output when run twice on the same data. Remember to set your random seeds in libraries like NumPy, PyTorch, or TensorFlow.

My solution has multiple files. How do I submit it?
You can include multiple files in your .zip submission (model weights, helper scripts, etc.). Just make sure your main solution.py file is at the root of the archive.

Winner's deliverables guide
Congratulations on your performance! To finalize your ranking and receive your prize, we need to verify your solution and understand your approach.

Our goal is to be able to reproduce your results from scratch and understand your methodology. Please provide a zip archive or a GitHub repository containing the following two main components:

1. Source Code
2. Technical Report
1. Source code & scripts
Your codebase should be clean, organized, and capable of reproducing your winning submission.

A. Environment setup
Dockerfile (preferred): A Dockerfile that builds the exact environment used for training and inference.
requirements.txt / environment.yml: A complete list of dependencies with pinned versions (e.g., torch==2.1.0, numpy==1.24.3).
Setup instructions: A brief README.md section on how to install dependencies and prepare the environment.
B. Training pipeline
We must be able to retrain your model from raw data.

train.py / train.sh: A single entry point script to run the full training pipeline.
Command example: python train.py --data_path ./datasets/train.parquet --output_dir ./weights
Configuration: Clearly expose key hyperparameters (batch size, learning rate, seq_len, hidden dims) in a config file (YAML/JSON) or as command-line arguments.
Reproducibility: Ensure all random seeds (Python, Numpy, PyTorch, etc.) are fixed. Retraining should yield the same model weights and score.
Hardware Requirements: In your README, state the hardware used for training (e.g., "Trained on 1x RTX 4090 for 4 hours" or "AWS p3.2xlarge").
C. Inference & submission
solution.py: The exact file used for your final submission.
Model Weights: The final trained model weights used in your submission.
2. Technical report (write-up)
Please provide a concise PDF or Markdown document (3-8 pages) covering the following. This is just as important as the code.

A. Executive summary
A brief overview of your solution (e.g., "A 4-layer Transformer with rotary embeddings and specific feature engineering").
B. Solution architecture
Model: Detailed description of the neural network architecture. Diagrams are highly encouraged.
Data Preprocessing: How did you handle the data? (Normalization, scaling, feature engineering, lag features, etc.).
Training Strategy:
Loss function used.
Optimizer and scheduler (warmup, decay).
Regularization techniques (Dropout, Weight Decay, etc.).
Validation strategy: How did you split the data?
C. Key boosters (critical steps)
The "Secret Sauce": Please explicitly list the 3-5 most important features, architectural decisions, or training tricks that gave you the biggest score boost.
Impact: If possible, estimate how much each improvement added to your score (e.g., "Feature X added +0.002 R²").
D. "What didn't work"
Describe approaches, architectures, or features you tried that failed or didn't improve the score.
Example: "I tried LSTM but it was 10% slower and less accurate than GRU."
Example: "Using feature X caused overfitting."
Summary checklist
Before submitting, please check:

[ ] README.md with clear instructions is included.
[ ] requirements.txt / Dockerfile allows building the env.
[ ] One-line command to start training is documented and working.
[ ] Random seeds are fixed for reproducibility.
[ ] Write-up includes the "Key Boosters" section.
[ ] Write-up includes the "What Didn't Work" section.
[ ] All model weights necessary for solution.py are included.
 
Independence: Each sequence is completely independent of the others. The market history from one sequence does not carry over to the next. When seq_ix changes, you are starting fresh.
Warm-up period: The first 100 steps (0-99) of every sequence are a "warm-up" period. You can use this data to build up your model's internal state (e.g., for an LSTM or Transformer), but you will not be scored on any predictions for these steps.
Scored predictions: Your score is based on predictions for steps 101 to 1000 (inclusive). These are the steps where need_prediction will be True.
Data ordering
Inside a sequence: Rows are always ordered chronologically.
step_in_seq 1 always comes after step_in_seq 0.
Between sequences: The sequences themselves are shuffled. seq_ix 10 is not related to seq_ix 11. This property is very useful for creating a reliable validation set.
TIP
How to create a validation set
Because the sequences are independent and shuffled, you can create a robust local validation set by splitting the data by seq_ix. For example, you can train your model on the first 80% of the sequences and test its performance on the remaining 20%.
Dataset sizes
Training set: The training data (train.parquet) contains approximately 500 sequences.
Test set: The hidden test set used for scoring is roughly the same size as the training set.
Evaluation metric
We evaluate predictions using the R² (coefficient of determination) score. For each feature i, the score is calculated as:

Feature Score Formula

The final score is the average of the R² scores across all N features:

Final Score Formula

A higher R² score is better.How to make a submission
This page covers the technical requirements for your submission, including the code format, how to package your files, and the resource limits.

What to submit
This is a code competition. You'll submit a .zip file containing all the code and artifacts needed to generate predictions.

The key requirements are:

The zip file must contain a solution.py file at its root.
Your solution.py must define a class named PredictionModel.
This class must have a predict(self, data_point) method.
The PredictionModel class
Here’s the required structure for your PredictionModel class:


import numpy as np
from utils import DataPoint

class PredictionModel:
    def __init__(self):
        # Initialize your model, load weights, etc.
        pass

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        # This is where your prediction logic goes.
        if not data_point.need_prediction:
            return None

        # When a prediction is needed, return a numpy array of length N.
        # Replace this with your model's actual output.
        prediction = np.zeros(data_point.state.shape)
        return prediction
Your predict method will receive a DataPoint object with the current market state. Your code should return None if need_prediction is False, and a NumPy array with your N feature predictions otherwise.

The DataPoint object comes from the provided utils.py file and has the following attributes:

seq_ix: int: The ID for the current sequence.
step_in_seq: int: The step number within the sequence.
need_prediction: bool: Whether a prediction is required for this point.
state: np.ndarray: The current market state vector of N features.
NOTE
Remember to handle the model's internal state.
When you encounter a new sequence (a new seq_ix), you must reset any recurrent state.
Including other files
Most solutions will need more than just solution.py. You can include other files in your zip archive, such as:

Model weights (e.g., .pt, .h5, .onnx files).
Helper Python modules (.py files).
Configuration files (e.g., .json, .yaml).
Small data files.
Just make sure solution.py is at the root of the archive.

How to package your solution
You need to package all your files into a single .zip archive.

macOS/Linux
Windows
Open a terminal.
cd into the directory that contains your solution files.
Run the following command:

zip -r ../submission.zip .
This creates submission.zip in the parent directory, containing everything from your current folder.

How submissions are scored
Docker
When you submit a solution, a scoring Docker container is deployed. The intricacies of the scoring process is not important here, but some details might be useful for you.

The container image is based on python:3.11-slim-bookworm
The environment variables used by some ML libraries are configured to prevent them from attempting to access the network
Below is the part of Dockerfile. You might want to use it locally for debug purposes.


FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get full-upgrade -y
RUN apt-get install -y curl libgomp1 p7zip-full build-essential && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
 && python -m pip install --prefer-binary --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt \
 && python -m pip install orbax-checkpoint \
 && python -m pip check && pip cache purge

# Keep heavy libs strictly offline at runtime 
ENV HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    WANDB_DISABLED=1 MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=false \
# Redirect all caches into /app
    HOME=/app \
    XDG_CACHE_HOME=/app/.cache \
    MPLCONFIGDIR=/app/.matplotlib \
    TORCH_HOME=/app/.cache/torch \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers \
    HF_DATASETS_CACHE=/app/.cache/huggingface/datasets \
    NUMBA_CACHE_DIR=/app/.cache/numba

### R̴̨̋E̴̟͝S̸̪̚T̸̢͘ ̶̜̈́I̴͖͗S̷̢͗ ̴̘̂C̶̕͜E̶̋͜N̵̼̓S̴̙͠O̴̘͐R̶̼͑Ě̵͕Ḓ̵̋ ###
Libs and packages
You may notice a requirements.txt is mentioned in Dockerfile above. We tried to install some reasonable set of popular libs often used in ML.

NOTE
If you need any package added to the scorer docker image — please drop us a line in Discord or email: get help
Here's the full requirements.txt:
Resource and time limits
Your submitted code will run in an isolated environment with the following constraints:

No internet access: The execution environment is offline.
Time limit: Your code must finish generating predictions for the entire test set in 60 minutes or less.
No GPU
CPU: 1 core; this is everything that is given in their website though the competition is over i am doing it for my github portfolio for recruiters; continue from where we left also analyzze what we did previously does it require modification i want to beat the 1st winner or atleast match them; don't rely on mudane coding assumptions that some method is better than other read modern reliable solutions and publications using web searches that would actually be more appropriate; the starter pack too is in your project folder; push my specs to max without crashing; i am working on my local machine; push claude opus 4.8 ultracode to it's max you don't mislead it; FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get full-upgrade -y
RUN apt-get install -y curl libgomp1 p7zip-full build-essential && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
 && python -m pip install --prefer-binary --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt \
 && python -m pip install orbax-checkpoint \
 && python -m pip check && pip cache purge

# Keep heavy libs strictly offline at runtime 
ENV HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    WANDB_DISABLED=1 MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=false \
# Redirect all caches into /app
    HOME=/app \
    XDG_CACHE_HOME=/app/.cache \
    MPLCONFIGDIR=/app/.matplotlib \
    TORCH_HOME=/app/.cache/torch \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers \
    HF_DATASETS_CACHE=/app/.cache/huggingface/datasets \
    NUMBA_CACHE_DIR=/app/.cache/numba

### R̴̨̋E̴̟͝S̸̪̚T̸̢͘ ̶̜̈́I̴͖͗S̷̢͗ ̴̘̂C̶̕͜E̶̋͜N̵̼̓S̴̙͠O̴̘͐R̶̼͑Ě̵͕Ḓ̵̋ ###; ### requirements.txt ###

# === Core numerics & IO ===
numpy>=1.26,<3
scipy>=1.13,<2
pandas>=2.1,<3
pyarrow>=15,<19
fastparquet>=2024.5.0
polars>=0.20,<1

# JAX CPU
jax[cpu]>=0.4.31

# === Utilities ===
tqdm>=4.66
joblib>=1.3
numba>=0.59,<1
einops>=0.7
rich>=13.7
loguru>=0.7
pydantic>=2.7,<3
hydra-core>=1.3,<2
omegaconf>=2.3,<3
pyyaml>=6.0
python-dotenv>=1.0

# === Classical ML ===
scikit-learn>=1.4,<2
xgboost>=2.0,<3
lightgbm>=4.3,<5
catboost>=1.2,<2
statsmodels>=0.14,<1

# === Deep Learning (PyTorch stack) ===
torch>=2.3,<3
torchvision>=0.18,<1
torchaudio>=2.3,<3
lightning>=2.4,<3           # (ex. pytorch-lightning)
torchmetrics>=1.4,<2
tensorflow>=2.17,<3           # provides tensorflow.keras

# === Transformers / seq modeling ===
transformers>=4.41,<5
accelerate>=0.30,<1
datasets>=2.19,<3
tokenizers>=0.15,<1
sentencepiece>=0.1.99

# === Experiment tracking & HPO ===
optuna>=3.5,<4
mlflow>=2.14,<3
wandb>=0.17,<1

# === Visualization ===
matplotlib>=3.8,<4
seaborn>=0.13,<1

# === Export / interop ===
onnxruntime>=1.18,<2
onnx>=1.16,<2

# === Testing ===
pytest>=8.0


# == EXTRAS ===
flax>=0.12.0
daal4py
