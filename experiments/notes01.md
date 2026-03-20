
#TF-IDF Experiment notes

- Logistic Regression and Linear SVC perform well even with unigram features `(1,1)` and do not show significant improvement with higher n-grams.
- Naive Bayes and LightGBM show improvement when using bigrams `(1,2)`, likely because they benefit from additional contextual information.
- Increasing the number of features (`max_features`) generally improves model performance across most models.
- RNN does not show improvement with changes in n-gram range. This is because TF-IDF input has sequence length = 1, so the RNN behaves like a dense layer without any temporal recurrence.
