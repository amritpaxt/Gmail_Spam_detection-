# Dataset

Place the CSV, TSV, Excel, or JSON dataset to train on in this folder. The included
pipeline automatically recognises common column names such as `Category`/`Message`,
`label`/`text`, `class`/`content`, and `spam`/`email`.

The expected target has two values: ham/legitimate and spam. The training command
also accepts an absolute path, so it can be run against a dataset elsewhere.

