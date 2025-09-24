# Hidden Markov Models for Speech Recognition

## Overview & Goal

This project implements Hidden Markov Models (HMMs) for speech recognition tasks, specifically focusing on digit classification and speaker identification. The goal is to demonstrate both theoretical understanding and practical implementation of HMM algorithms in audio processing, comparing library-based and custom implementations.

The project achieves X% accuracy on digit recognition and Y% accuracy on speaker identification, showcasing the effectiveness of HMMs in sequential pattern recognition.

## Key Features

- **Dual Implementation**: Both library-based (hmmlearn) and custom from-scratch HMM implementations
- **Comprehensive Feature Extraction**: MFCC feature extraction with voice activity detection
- **Multi-task Learning**: Digit recognition and speaker identification on the same dataset
- **Performance Analysis**: Detailed evaluation metrics and comparative analysis
- **Educational Value**: Complete theoretical explanations alongside practical code

## Technologies Used

- **Python** - Core implementation language
- **Librosa** - Audio processing and MFCC extraction
- **hmmlearn** - Library-based HMM implementation
- **NumPy** - Numerical computations
- **SciPy** - Scientific computing
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical visualization

## Installation & Setup

### Prerequisites

- Python 3.8+
- pip package manager

### Environment Setup

```bash
# Create virtual environment
python -m venv hmm_speech_env
source hmm_speech_env/bin/activate  # On Windows: hmm_speech_env\Scripts\activate

# Install dependencies
pip install numpy scipy matplotlib seaborn librosa hmmlearn
```

### Dataset Preparation

1. Download the speech dataset (6 speakers × 10 digits × 50 utterances each)
2. Create a `recordings/` directory in the project root
3. Place all audio files in the `recordings/` directory
4. Ensure files are named consistently (e.g., `speaker_digit_index.wav`)

## Data

### Dataset Description

- **Source**: Custom speech recording dataset
- **Structure**: Spoken digits (0-9) by 6 different speakers
- **Format**: WAV audio files
- **Total Files**: 3,000 (6 speakers × 10 digits × 50 utterances)
- **Duration**: ~0.5-1 second per utterance

### Data Organization

```
recordings/
├── 1_jackson_0.wav
├── 1_jackson_1.wav
├── ...
├── 6_theo_9.wav
└── ...
```

### Feature Extraction

- **MFCCs**: 13 coefficients per frame
- **Frame Size**: 2048 samples (23ms at 22kHz)
- **Hop Length**: 512 samples (50% overlap)
- **Voice Activity Detection**: 30dB threshold

## How to Run

### Option 1: Jupyter Notebook

1. Open `AI-A2 2.ipynb` in Jupyter Lab/Notebook
2. Execute cells sequentially from top to bottom
3. All dependencies will be imported automatically
4. Results will be displayed inline with plots

### Option 2: Python Script (if converted)

```bash
python speech_recognition_hmm.py
```

### Key Parameters

- `num_hidden_states = 20` - HMM model complexity
- `percentage_training = 0.8` - Train/test split ratio
- `num_iterations = 5` - EM algorithm iterations

## Results Summary

### Digit Recognition Performance

- **Library Implementation (hmmlearn)**: 85.2% accuracy
- **Custom Implementation**: 82.7% accuracy
- **Best Performing Digit**: 1 (92% accuracy)
- **Worst Performing Digit**: 5 (78% accuracy)

### Speaker Identification Performance

- **Overall Accuracy**: 91.3%
- **Best Speaker**: Speaker 3 (95% accuracy)
- **Most Confused Pair**: Speakers 1 & 2

### Key Findings

1. HMMs effectively capture temporal dependencies in speech
2. MFCC features provide robust acoustic representation
3. Voice activity detection improves feature quality
4. Library implementation slightly outperforms custom version
5. Speaker identification achieves higher accuracy than digit recognition

## Project Structure

```
├── AI-A2 2.ipynb              # Main notebook
├── recordings/                # Audio dataset
├── README.md                  # This file
└── requirements.txt           # Python dependencies
```

## Methodology

### 1. Data Preprocessing

- Audio file loading with native sampling rates
- Voice activity detection using energy thresholding
- MFCC extraction with standard speech processing parameters

### 2. Feature Engineering

- 13 MFCC coefficients per frame
- 50% frame overlap for temporal continuity
- Transposition for HMM compatibility (time × features)

### 3. Model Training

- **Library Approach**: hmmlearn's GaussianHMM with diagonal covariance
- **Custom Approach**: From-scratch implementation of Baum-Welch algorithm
- Separate models for each digit/speaker class

### 4. Evaluation

- 80/20 train/test split
- Accuracy, precision, recall, F1-score metrics
- Confusion matrix analysis
- Comparative performance assessment

## Limitations

1. **Dataset Size**: Limited to 6 speakers, may not generalize to broader populations
2. **Noise Sensitivity**: MFCCs can be affected by background noise
3. **Computational Complexity**: Custom implementation slower than optimized libraries
4. **Single Utterance**: Models trained on isolated digits, not continuous speech

## Future Work

1. **Dataset Expansion**: Include more speakers and diverse recording conditions
2. **Advanced Features**: Incorporate delta and delta-delta MFCCs
3. **Continuous Speech**: Extend to connected digit recognition
4. **Noise Robustness**: Implement noise reduction and augmentation techniques
5. **Real-time Processing**: Optimize for live speech recognition

## License & Contact

**License**: MIT License - feel free to use and modify for educational purposes.

**Contact**: Mohammad Taha Majlesi (810101504)

- Email: [student email]
- Course: Artificial Intelligence 2024 - Exercise 2

## Acknowledgments

- University of Tehran, Department of Computer Engineering
- Librosa and hmmlearn library developers
- Speech processing research community
