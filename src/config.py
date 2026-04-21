MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
TEST_SIZE = 0.1
RANDOM_SEED = 42

DATA_PATH = "../data/raw/NewsCategorizer.csv"
OUTPUT_DIR = "../saved_models/bert-base-uncased"
LOG_DIR = "../logs"