import torch

# --------------------------------------------------------------
# ALGORITHM CONFIGURATION
# Choose the algorithm to use
# Options: "Naives_bayes", "Perceptron", "MLP_H1", "MLP_H2"
# --------------------------------------------------------------
ALGORITHM = "MLP_H1"

# --------------------------------------------------------------
# MYSQL LOGIN CREDENTIALS
# Provide MySQL database connection credentials
# --------------------------------------------------------------
USER = "optuna_seet"
PASSWORD = "@g3NYkke*eAFRs"
DATABASE_NAME = "optuna_MLP"

# --------------------------------------------------------------
# ENDPOINT CONFIGURATION
# Define the endpoint for Optuna (Local or Ngrok)
# --------------------------------------------------------------
ENDPOINT = "localhost"  # Local usage
# ENDPOINT = "1.tcp.eu.ngrok.io:3791"  # Example usage with Ngrok (LOTR)

# --------------------------------------------------------------
# DATA PATH CONFIGURATION
# Specify the folder containing the data
# --------------------------------------------------------------
DATA_PATH = "./data/"

# --------------------------------------------------------------
# OPTIMIZATION PARAMETERS
# --------------------------------------------------------------

# Number of trials for optimization
N_TRIALS = 5

# Toggle Weights & Biases integration (only works with neural networks)
WANDB_ACTIVATE = False

# Output files for hyperparameters
OUTPUT_HP_FILENAME = f"hp_{ALGORITHM}"
OUTPUT_HP_PATH = "./hyperparameters"

# --------------------------------------------------------------
# PREDICTION SETTINGS
# --------------------------------------------------------------

# Input files for hyperparameters used for predictions
INPUT_HP_FILENAME = f"hp_{ALGORITHM}"
INPUT_HP_PATH = "./hyperparameters"

# Output files for predictions
PREDICTION_FILENAME = f"{ALGORITHM}_pred"
PREDICTION_PATH = "./output"

# --------------------------------------------------------------
# INTERNAL VARIABLES (DO NOT MODIFY)
# --------------------------------------------------------------
INPUTS_DOCUMENTS = None
LABELS_DOCUMENTS = None
TEST_DOCUMENTS = None
VOCAB = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_PATH = ""
