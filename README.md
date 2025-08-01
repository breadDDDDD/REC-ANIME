# REC-ANIME🎌

REC-ANIME, Deep Learning Collaborative Filtering Recommendation System for Animes

## 📌 Features

- Matrix factorization with user-anime embeddings
- Cosine similarity for finding similar users and animes outside the training set.
- Custom Keras training pipeline with early stopping and learning rate scheduler.
- Utility functions to analyze preferences and generate recommendations

## 🧠 Model Architecture

The model uses two embedding layers for users and animes, computes their dot product, and passes it through a dense layer with batch normalization and sigmoid activation.

## 📊 Data
The dataset consists of user ratings for animes and additional metadata like genres and titles. It is preprocessed to create user and anime embeddings.
For data source, please see the Kaggle dataset [here](https://www.kaggle.com/datasets/breaddddd/anime-list-cleaned)

## 🚀 Inference
Currently, the model supports two main inference tasks:
1. **Finding Similar Animes**: Given an anime ID, it retrieves the top N animes.
2. **User based Recommendations**: Given a user ID, it recommends the top N animes based on the user's preferences.

## 📚 Usage
Currently the model can be used by running the inference.py file.
ou can run inference with:

```bash
python inference.
