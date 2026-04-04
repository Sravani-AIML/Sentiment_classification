# Movie Review Sentiment_classification

A comprehensive study of sentiment analysis on 50,000 IMDB movie reviews using three distinct architectural approaches.

## Project Structure
- `data/`: (Local only) Raw and processed datasets.
- `src/`: Core Python modules for preprocessing, vectorization, tokenization and modeling.
- `experiments/`: 
  1. **TF-IDF + Linear Models + RNN**: Traditional NLP baseline.
  2. **Custom Embeddings + LSTM/GRU/CNN**: Deep learning with LSTMs and Convolutions.
  3. **Transfer Learning + CNN**: Using pre-trained embeddings (Word2Vec/GloVe) with CNN architectures.
- `results/`: Performance metrics and visualization plots.

## Experiment 1: TF-IDF and Linear Models
In this initial experiment, I used TF-IDF Vectorization to convert movie reviews into numerical features. My goal was to establish a strong baseline before moving on to more complex deep learning architectures.

Key Observations:
- Linear Model Performance: Logistic Regression and Linear SVC performed exceptionally well. Because TF-IDF produces highly sparse, high-dimensional matrices, linear models are often more efficient at finding a clear separating hyperplane than complex non-linear models.

- The Baseline: Logistic Regression with a simple unigram ngram_range=(1, 1) provided a solid baseline accuracy of 88.53%.

- Context Matters (N-grams): Models like Naive Bayes and LightGBM saw accuracy improvements when moving to trigrams ngram_range=(1, 3). This suggests that their probability predictions rely heavily on local word context and phrases rather than just individual words.

- TF-IDF with RNNs: I tested an RNN (SimpleRNN with ReLU) on these TF-IDF vectors, but it only reached ~87%. Since TF-IDF vectors do not contain intrinsic semantic sequence information (unlike embeddings), the RNN essentially behaved like a standard Dense layer; there was no meaningful temporal "recurrence" for the model to learn from.

- Feature Scaling: Increasing the max_features from 5,000 to 10,000 led to a noticeable jump in accuracy (peaking at 89.19%), showing that a larger vocabulary helps the model capture more specific sentiment-carrying tokens.

### Plot represents the tf-idf results with accuracy> 87
<img width="3600" height="2100" alt="tfidf_top_results (1)" src="https://github.com/user-attachments/assets/4861d933-692a-4bf8-80f4-9e69cf09675e" />

This experiment proved that a well-tuned Logistic Regression model on a large TF-IDF vocabulary is a very difficult baseline to beat, providing high accuracy with almost near-instant training times.

## Experiment 2: Deep Learning with Custom Embeddings
In this phase, I moved beyond frequency-based vectors (TF-IDF) to Trainable Word Embeddings. This allows the model to learn mathematical relationships between words based on their context in the IMDB dataset. I compared various architectures including LSTM, GRU, and CNN.

Key Findings:

- CNN Performance: The CNN architecture was the standout performer. By using sliding windows (kernels), it effectively captured local patterns and n-grams within the reviews. It achieved higher accuracy than the standard RNN/LSTM models while being significantly faster to train.

- The Power of Sequence Length: Increasing the sequence_length had the most dramatic impact on accuracy. As the model was allowed to "read" more of the movie review (moving from 100 to 250 words), accuracy increased significantly. This suggests that for sentiment analysis, the context found in the middle or end of a review is often crucial for a correct classification.

- Diminishing Returns of Embedding Size: Surprisingly, increasing the embedding_size from 50 to 100 provided only a marginal boost in accuracy. This indicates that a 50-dimensional space was already sufficient to capture the semantic relationships of our vocabulary for this specific task.

- Vocabulary Size: Expanding the vocab_size from 10,000 to 15,000 words allowed the model to recognize more "long-tail" sentiment words, resulting in a steady increase in final accuracy, peaking at 89.07%.

- Hybrid Models: I also tested a CNN-GRU hybrid. While it was computationally heavy (taking over 5,000 seconds), it did not outperform the standalone CNN, suggesting that for this dataset, simple spatial pattern recognition (CNN) is more effective than combining it with long-term dependency tracking (GRU). 

### sequence vs embedding vs accuracy

<img width="1000" height="600" alt="seq_vs_emb" src="https://github.com/user-attachments/assets/c8dbaafe-4642-4f5a-a535-d7404be0baac" />

Sequence vs. Embedding Size: Shows the steep climb in accuracy as we include more words per review.

### vocab vs accuracy

<img width="700" height="600" alt="vocab_vs_acc" src="https://github.com/user-attachments/assets/763ed831-5cb5-47a4-a5a5-03e0073f61ff" />

Vocabulary Expansion: Demonstrates the benefit of keeping more unique tokens in the word index.

## Experiment 3: Transfer Learning with Pre-trained Embeddings
In this final phase, I compared two of the most popular pre-trained word embedding techniques—Word2Vec and GloVe—using a CNN architecture. The goal was to see if "Transfer Learning" (using embeddings trained on massive external datasets) could outperform models trained from scratch on our 50,000 reviews.

Top Performance: Word2Vec emerged as the overall winner with an accuracy of 89.60%.

Word2Vec vs. GloVe: Word2Vec slightly outperformed GloVe (89.12%). This is often because Word2Vec's predictive approach (Skip-gram or CBOW) can capture more nuanced syntactic and semantic relationships that are particularly useful for the specific vocabulary used in movie reviews.

Generalization Power: Both pre-trained models provided a more robust feature space than the custom embeddings from Experiment 2, as they come with "pre-learned" knowledge of word similarities from much larger corpora (like Google News or Wikipedia).

#### Computational Cost vs. Accuracy:

<img width="1200" height="700" alt="final_models_comapar" src="https://github.com/user-attachments/assets/1942ba87-8098-46e6-a713-197986481626" />

While Word2Vec gave the best results, it came with a significant training time of approximately 2,836 seconds.

When looking at the Accuracy vs. Computational Cost plot, Logistic Regression remains an incredibly impressive competitor. It achieved 89.19% (only 0.4% less than Word2Vec) in just 1.35 seconds.




