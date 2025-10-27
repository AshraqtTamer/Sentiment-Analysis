# 🎬 IMDB Movie Review Sentiment Analysis using LSTM

## 🧠 Project Overview
This project focuses on performing **sentiment analysis** on movie reviews using **deep learning (LSTM)**.  
The dataset used is the **IMDB Dataset** from Kaggle, which contains 50,000 reviews labeled as **positive** or **negative**.

Using **TensorFlow / Keras**, the model learns to understand the context of words and predict the overall sentiment of a review.

---

## 📊 Dataset
**Dataset Name:** [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

- **Source:** Kaggle  
- **Size:** 50,000 reviews  
- **Columns:**
  - `review` — Text of the movie review.
  - `sentiment` — Label indicating sentiment (`positive` / `negative`).

Example:

| review | sentiment |
|---------|------------|
| "A wonderful little production..." | positive |
| "Basically there's a family where a little boy..." | negative |

---

## ⚙️ Technologies Used
- **Python 3.x**
- **TensorFlow / Keras**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**

---

## 🚀 Project Workflow

### 1. Importing Libraries
Essential Python libraries for data processing, visualization, and deep learning are imported.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
