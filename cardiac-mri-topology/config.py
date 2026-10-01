import torch

# Data Configuration
TARGET_SIZE = (256, 256)
BATCH_SIZE = 16
NUM_WORKERS = 2

# Architecture Configuration
IN_CHANNELS = 1
NUM_SEG_CLASSES = 4   # Background, LV, RV, MYO
NUM_DISEASE_CLASSES = 5 # Normal, MINF, DCM, HCM, ARV

# Optimization Hyperparameters
LEARNING_RATE = 1e-4
EPOCHS = 10

# Loss Balancing Weights (L_total = alpha*L_seg + beta*L_class + gamma*L_topo)
ALPHA = 1.0
BETA = 0.5
GAMMA = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
