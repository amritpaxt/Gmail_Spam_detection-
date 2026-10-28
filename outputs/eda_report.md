# Mail spam dataset: data-quality and EDA report

## Data loading and cleaning

| Measure | Value |
|---|---:|
| Source text column | `Message` |
| Source label column | `Category` |
| Input rows | 5,573 |
| Blank messages removed | 0 |
| Invalid/non-class rows removed | 1 |
| Duplicate messages removed | 415 |
| Clean rows | 5,157 |
| Ham | 4,516 |
| Spam | 641 |

## Message and word-length insights

| Class | Mean message characters | Median characters | Mean words | Median words |
|---|---:|---:|---:|---:|
| Ham | 70.78 | 53.00 | 14.71 | 11.00 |
| Spam | 137.12 | 148.00 | 25.18 | 27.00 |

`eda_overview.png` visualises class balance, message length, word count, and average word length.
`top_words.png` compares the most frequent non-stopwords in each class.
