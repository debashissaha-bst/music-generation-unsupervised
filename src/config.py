from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw_midi"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_SPLIT = PROJECT_ROOT / "data" / "train_test_split"
OUTPUTS_MIDI = PROJECT_ROOT / "outputs" / "generated_midis"
OUTPUTS_PLOTS = PROJECT_ROOT / "outputs" / "plots"
OUTPUTS_SURVEY = PROJECT_ROOT / "outputs" / "survey_results"
CHECKPOINTS = PROJECT_ROOT / "outputs" / "checkpoints"

# Preprocessing
STEPS_PER_QUARTER = 4  
MAX_SEQ_LEN = 128
MAX_DURATION_STEPS = 32
TRAIN_RATIO = 0.85
RANDOM_SEED = 42

# Genre ids 
GENRE_NAMES = ["classical", "jazz", "rock", "pop", "electronic"]
NUM_GENRES = len(GENRE_NAMES)

# Models
EMBED_DIM = 256
HIDDEN_DIM = 512
LATENT_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.1
TRANSFORMER_HEADS = 8
TRANSFORMER_LAYERS = 4
TRANSFORMER_FF = 1024
MAX_SEQ_TRANSFORMER = 256

# Training
BATCH_SIZE = 32
LR = 1e-3
EPOCHS_AE = 30
EPOCHS_VAE = 40
EPOCHS_TRANSFORMER = 50
EPOCHS_RLHF = 20
KL_BETA = 0.5
KL_ANNEAL_EPOCHS = 10

# RLHF
RL_LR = 1e-5
RL_BATCH = 8
RL_SAMPLES_PER_STEP = 16

for p in (
    DATA_RAW,
    DATA_PROCESSED,
    DATA_SPLIT,
    OUTPUTS_MIDI,
    OUTPUTS_PLOTS,
    OUTPUTS_SURVEY,
    CHECKPOINTS,
):
    p.mkdir(parents=True, exist_ok=True)
