
## TF-IDF Vectors with Linear Models vs Recurrent Neural Network
- Logistic Regression and Linear SVC perform well even with unigram features (1,1) and do not show significant improvement with higher n-grams.
- Naive Bayes and LightGBM show improvement when using bigrams (1,2), likely because they benefit from additional contextual information.
- Increasing the number of features (max_features) generally improves model performance across most models.
- RNN does not show improvement with changes in n-gram range. This is because TF-IDF input has sequence length = 1, so the RNN behaves like a dense layer without any temporal recurrence.

## Embedding with different neural networks architecures
- Compared to LSTM, GRU performed better with slightly higher accuracy and less training time.
- After observing GRU performance, BiGRU was tried, but accuracy decreased compared to both GRU and LSTM, and training time almost doubled. So, bidirectional complexity did not help much for this task.
- The CNN performed similarly to the GRU but required significantly less training time; additionally, it effectively captured bigram and trigram features through its kernels.
- Based on this, more experiments were done on CNN with different parameters.
- Increasing sequence length consistently improved accuracy, as more context from the reviews was captured.
- Increasing embedding size resulted in only minimal improvement compared to sequence length.

## Transfer Learning with pretrained embedding models
- Word2Vec outperformed the pre-trained GloVe 100d because it was trained directly on the movie review dataset, allowing it to capture domain-specific sentiment and vocabulary nuances. In contrast, the GloVe embeddings relied on global co-occurrence statistics from general corpora like Wikipedia, which lacked the specialized context necessary for high-accuracy sentiment classification in this specific domain.



