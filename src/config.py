import torch

# DON'T CHANGE

# Data
INPUTS_DOCUMENTS = None
LABELS_DOCUMENTS = None
TEST_DOCUMENTS = None

WANDB_ACTIVATE = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MYSQL et OPTUNA
USER = ""
PASSWORD = ""
DATABASE_NAME = ""
ENDPOINT = ""

# HYPERPARAMETERS
OUTPUT_HP_FILENAME = ""
OUTPUT_HP_PATH = ""
