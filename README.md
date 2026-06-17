# Lane-Change-Intention-LSTM
Encoder-Decoder LSTM for predicting lane change intentions of surrounding vehicles using NGSIM highway trajectory data — 85.3% accuracy
# Lane Change Intention Identification Using Encoder-Decoder LSTM

## Overview
A deep learning system that predicts the driving intention of surrounding vehicles on highways —
classifying each vehicle as going **Straight**, turning **Left**, or turning **Right** —
using only historical trajectory data from the NGSIM dataset.

Inspired by the probabilistic risk assessment framework proposed in:
> Huang et al., "A probabilistic risk assessment framework considering lane-changing behavior interaction," *Science China Information Sciences*, 2020. https://doi.org/10.1007/s11432-019-2983-0

Many implementation details were not specified in the paper and were designed independently.

---

## Problem
Predicting the lane-change intentions of surrounding vehicles in advance is critical for safe
autonomous driving. This system classifies vehicle intentions 3 seconds before the maneuver
occurs, giving an autonomous vehicle enough time to react.

---

## Approach

### Architecture — Encoder-Decoder LSTM
- **Encoder LSTM** processes 3 seconds of historical trajectory (15 frames at 5 Hz)
- **Decoder LSTM** produces the intention classification
- 4 stacked LSTM layers, 128 hidden units, dropout 0.2

### Input Features (23 features per frame)
Each frame includes the ego vehicle state plus surrounding vehicle states:
- Ego: position (x, y), velocity
- 6 surrounding vehicles (Front, Back, Left-Front, Left-Rear, Right-Front, Right-Rear): relative position (dx, dy), velocity
- Lane flags: left lane available, right lane available

### Threshold-Based Decision Logic
Inspired by the paper's confidence thresholds:
- Left lane change: confidence ≥ 80%
- Right lane change: confidence ≥ 80%
- Straight: confidence ≥ 70%
- Fallback to argmax if no threshold met

### Training Details
- Optimizer: Adam (lr=5e-4, decay=0.9)
- Loss: CrossEntropyLoss
- Batch size: 32
- Early stopping (patience=7)
- Gradient clipping (max norm=1.0)

---

## Dataset
**NGSIM (Next Generation Simulation)** — US-101 and I-80 highway sections  
Real vehicle trajectory data at 10 Hz, downsampled to 5 Hz.  
Download: https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm

**Preprocessing pipeline:**
- Filtered to mainline lanes only (lanes 1–5)
- Removed short tracks (< 8 seconds)
- Downsampled from 10 Hz to 5 Hz
- Smoothed with 3-frame rolling window
- Balanced sampling: 5,000 samples per class (15,000 total)
- Split: 70% train / 10% validation / 20% test

---

## Results

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Straight | 95.9% | 92.1% | 93.9% | 1000 |
| Left | 83.6% | 74.2% | 78.6% | 1000 |
| Right | 77.8% | 89.6% | 83.3% | 1000 |
| **Overall** | **85.8%** | **85.3%** | **85.3%** | 3000 |

**Confusion Matrix:**
```
              Predicted
              Straight  Left  Right
Actual Straight  921     58    21
       Left       23    742   235
       Right      16     88   896
```

---

## Stack
- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

---

## Project Structure
```
├── preprocessing_whole_data_v2.py   # Raw NGSIM CSV cleaning and downsampling
├── NGSIM_LABEL_FULL_V2.py          # Feature extraction and intention labeling
├── dataset_split_v2.py             # Balanced train/val/test splitting
├── LSTM_TRAIN_V2.py                # Model training with validation
└── LSTM_TEST.py                    # Model evaluation and visualization
```

---

## How to Run

```bash
# Install dependencies
pip install torch numpy pandas scikit-learn matplotlib

# Step 1 - Preprocess raw NGSIM data
python preprocessing_whole_data.py

# Step 2 - Extract features and labels
python NGSIM_LABEL_FULL.py

# Step 3 - Split dataset
python dataset_split.py

# Step 4 - Train model
python LSTM_TRAIN.py

# Step 5 - Evaluate
python LSTM_TEST.py
```

---

## Author
Nada Selim — Computer Vision & ML Engineer  
[LinkedIn](https://www.linkedin.com/in/nada-selim-phd-student-a053b5223) | [GitHub](https://github.com/nadaselim-AI)
 
Nada Selim — Computer Vision & ML Engineer  
[LinkedIn](https://www.linkedin.com/in/nada-selim-phd-student-a053b5223) | [GitHub](https://github.com/nadaselim-AI)
