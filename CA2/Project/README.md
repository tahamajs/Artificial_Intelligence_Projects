# Hidden Markov Models for Speech Recognition

## Project Overview

This project implements a comprehensive Hidden Markov Model (HMM) based speech recognition system capable of both digit recognition and speaker identification. The implementation includes both a custom from-scratch HMM class and comparisons with established libraries, providing a complete educational and research resource.

## Features

### 🔬 Complete HMM Implementation

- **Custom HMM Class**: Full implementation of forward-backward algorithms, Baum-Welch training, and Gaussian emissions
- **Mathematical Validation**: Performance comparable to hmmlearn library
- **Modular Architecture**: Clean, well-documented code for educational purposes

### 🎵 Audio Processing Pipeline

- **MFCC Extraction**: Mel-frequency cepstral coefficients with voice activity detection
- **Data Preprocessing**: Systematic organization and validation of audio datasets
- **Feature Engineering**: Robust preprocessing for optimal HMM performance

### 📊 Comprehensive Evaluation

- **Multi-Task Support**: Digit recognition (0-9) and speaker identification (6 speakers)
- **Performance Metrics**: Accuracy, precision, recall, F1-score analysis
- **Comparative Studies**: Direct comparison between custom and library implementations

### 📚 Educational Resources

- **Detailed Documentation**: Extensive Markdown explanations throughout
- **Theoretical Foundation**: Clear explanations of HMM mathematics and speech recognition
- **Practical Analysis**: Interpretation of results and performance patterns

## Installation

### Prerequisites

```bash
Python 3.7+
```

### Dependencies

```bash
pip install numpy scipy librosa hmmlearn matplotlib seaborn
```

### Dataset

The project uses audio data for digits 0-9 spoken by multiple speakers. Ensure audio files are properly organized in the expected directory structure.

## Usage

### 1. Data Preparation

```python
# Load and preprocess audio data
mfccs_list = load_and_preprocess_audio()
```

### 2. Feature Extraction

```python
# Extract MFCC features with VAD
mfcc_features = extract_mfcc_with_vad(audio_file)
```

### 3. Model Training

```python
# Train custom HMM
hmm_model = HMM(num_hidden_states=6)
hmm_model.train(training_data, num_iterations=10)

# Or use library implementation
from hmmlearn import hmm
model = hmm.GaussianHMM(n_components=6)
model.fit(training_data)
```

### 4. Evaluation

```python
# Evaluate performance
accuracy = evaluate_model(model, test_data)
print(f"Accuracy: {accuracy:.3f}")
```

## Project Structure

```
├── AI-A2 2.ipynb          # Main implementation notebook
├── README.md              # Project documentation
├── CA2/
│   ├── Description/       # Project requirements and solutions
│   └── Project/          # Implementation files
└── Data/                 # Audio datasets (if included)
```

## Key Components

### Custom HMM Class

- **Forward Algorithm**: Computes observation likelihoods
- **Backward Algorithm**: Computes backward probabilities
- **Baum-Welch Training**: Expectation-Maximization for parameter learning
- **Gaussian Emissions**: Multivariate Gaussian probability density functions

### Audio Processing

- **MFCC Extraction**: 13-dimensional feature vectors
- **Voice Activity Detection**: Removes silence and noise
- **Data Concatenation**: Combines multiple utterances for training

### Evaluation Framework

- **Confusion Matrices**: Detailed error analysis
- **Performance Metrics**: Comprehensive statistical evaluation
- **Comparative Analysis**: Custom vs. library implementations

## Results

### Digit Recognition

- Custom HMM implementation achieves competitive performance
- Performance varies by digit due to acoustic similarities
- Confusion matrix reveals common misclassification patterns

### Speaker Identification

- Effective speaker discrimination using HMMs
- Performance analysis across different speakers
- Validation of model generalization capabilities

## Technical Details

### HMM Parameters

- **Hidden States**: 6 states optimized for speech segments
- **Emission Model**: Gaussian distributions with diagonal covariance
- **Training**: Baum-Welch algorithm with configurable iterations

### Audio Specifications

- **Sample Rate**: Standard audio processing rates
- **Frame Length**: 25ms analysis windows
- **Frame Shift**: 10ms overlap between frames

## Educational Value

This implementation serves as a comprehensive resource for:

- Understanding HMM mathematical foundations
- Learning speech recognition principles
- Implementing machine learning algorithms from scratch
- Analyzing model performance and limitations

## Future Enhancements

- **Advanced Features**: Integration of delta and delta-delta coefficients
- **Model Optimization**: Hyperparameter tuning and architecture improvements
- **Real-time Processing**: Live audio stream analysis capabilities
- **Deep Learning Integration**: Hybrid HMM-neural network approaches

## References

1. Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition.
2. Davis, S., & Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition.
3. Huang, X., et al. (2001). Spoken Language Processing: A Guide to Theory, Algorithm and System Development.

## License

This project is developed for educational purposes as part of an Artificial Intelligence course assignment.

## Contact

For questions or contributions, please refer to the course materials or contact the development team.
