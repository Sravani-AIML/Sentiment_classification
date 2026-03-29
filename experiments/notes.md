
#TF-IDF Experiment notes
- Logistic Regression and Linear SVC perform well even with unigram features (1,1) and do not show significant improvement with higher n-grams.
- Naive Bayes and LightGBM show improvement when using bigrams (1,2), likely because they benefit from additional contextual information.
- Increasing the number of features (max_features) generally improves model performance across most models.
- RNN does not show improvement with changes in n-gram range. This is because TF-IDF input has sequence length = 1, so the RNN behaves like a dense layer without any temporal recurrence.

#Embedding Experiment notes
- Compared to LSTM, GRU performed better with slightly higher accuracy and less training time.
- After observing GRU performance, BiGRU was tried, but accuracy decreased compared to both GRU and LSTM, and training time almost doubled. So, bidirectional complexity did not help much for this task.
- CNN gave performance similar to GRU but with significantly less training time.
- Based on this, more experiments were done on CNN with different parameters.
- Increasing sequence length consistently improved accuracy, as more context from the reviews was captured.
- Increasing embedding size resulted in only minimal improvement compared to sequence length.


