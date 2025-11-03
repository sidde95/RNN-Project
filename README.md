# Twitter & Reddit Sentiment Analysis using Simple RNN

This project performs **Sentiment Analysis** on text data collected from **Twitter** and **Reddit**, using a **Recurrent Neural Network (RNN)** built with **TensorFlow/Keras**.  
The model classifies text into **Negative (-1)**, **Neutral (0)**, and **Positive (1)** sentiments.

---

## Project Overview

The project demonstrates how **Deep Learning for NLP** can extract sentiment information from real-world social media data.  
It follows an **end-to-end pipeline** from data cleaning and tokenization to model training and evaluation.

**Goal:**  
To build a deep learning model capable of understanding sentiment trends in political tweets and comments, enabling real-time public opinion monitoring.

---

## Key Features

- Cleaned and preprocessed multi-source text data (Twitter + Reddit)  
- Applied **Lemmatization** and regex-based text normalization  
- Converted text to numerical representation using **One-Hot Encoding**  
- Used **Padding** to maintain equal input sequence lengths  
- Trained a **Simple RNN** with ELU activation and EarlyStopping  
- Evaluated model on unseen Reddit dataset for generalization  
- Achieved strong sentiment classification performance

---

## Model Architecture

| Layer Type  | Description |
|--------------|-------------|
| Embedding Layer | Converts word indices into dense vector representations |
| SimpleRNN (128 units, ELU) | Captures sequential text dependencies |
| Dense (Softmax) | Outputs 3-class sentiment probabilities |

**Loss Function:** Sparse Categorical Crossentropy  
**Optimizer:** Adam  
**EarlyStopping:** Patience = 20 epochs  

---

## Dataset

Dataset Source: [Twitter and Reddit Sentimental Analysis Dataset – Kaggle](https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset)

| Platform | Records | Columns | Description |
|-----------|----------|----------|-------------|
| Twitter | ~163,000 | 2 | Cleaned tweets with sentiment labels |
| Reddit | ~37,000 | 2 | Cleaned comments with sentiment labels |

**Sentiment Labels:**
- `-1` → Negative  
- `0` → Neutral  
- `1` → Positive  

---

## Data Preprocessing Steps

1. **Remove URLs, mentions, and special characters** using regex  
2. **Convert text to lowercase** for normalization  
3. **Tokenize and Lemmatize** words using NLTK’s WordNetLemmatizer  
4. **Encode text** using One-Hot representation  
5. **Pad sequences** to match maximum sentence length (max_len = 52)  
6. **Split data** into training and validation sets (80:20)  

---

## Model Training

```python
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_len))
model.add(SimpleRNN(128, activation='elu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---
## Model Evaluation

| Metric                      | Score                       |
| --------------------------- | --------------------------- |
| Accuracy on Reddit Test Set | ~0.85                       |
| Classification Metric       | Precision, Recall, F1-Score |

---

##Tech Stack

Languages & Frameworks
- Python 3.10
- TensorFlow / Keras

Libraries

- pandas, numpy
- nltk (WordNetLemmatizer)
- scikit-learn (train_test_split, classification_report)
- matplotlib, seaborn (visualization)
- re (regex cleaning)

---
## Insights

- The model effectively distinguishes emotional polarity in both Twitter and Reddit content.
- RNN architecture successfully captures sequential dependencies in user text.
- Applying early stopping prevents overfitting and improves generalization.
- Demonstrates portability — trained on Twitter, tested on Reddit.
