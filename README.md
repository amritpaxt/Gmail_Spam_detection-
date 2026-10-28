# Mail Spam Detection with Gmail Labels

An end-to-end, reproducible spam classifier for the supplied `Category` / `Message`
dataset, plus a local Gmail watcher that applies custom labels. The model, OAuth
token, and generated reports stay on your laptop.

> **Accuracy warning:** the supplied data is SMS-style message data, not your
> personal Gmail data. The measured 98.93% accuracy below applies to the
> supplied dataset only. Gmail accuracy has not been measured because your
> inbox messages do not have verified ground-truth labels.

## Verified results on the supplied training dataset

The model was trained from `C:\Users\Admin\OneDrive\Desktop\MailSpam\email.csv`.
After cleaning and exact-message de-duplication, 5,157 records remained: 4,516
ham and 641 spam. Linear SVC was selected as the best candidate.

| Metric | Result |
|---|---:|
| Five-fold validation accuracy | 99.13% |
| Five-fold validation spam F1 | 96.38% |
| Hold-out test accuracy | 98.93% |
| Hold-out spam precision | 100.00% |
| Hold-out spam recall | 91.41% |
| Hold-out spam F1 | 95.51% |

It includes the full requested workflow:

1. Data loading, schema validation, label normalisation, blank-row removal, and de-duplication.
2. Exploratory data analysis and class balance.
3. Character/message-length analysis.
4. Average word-length analysis.
5. Word-count distributions and frequent-word insights.
6. Text normalisation (lowercasing, URL/email/phone tokens, whitespace cleanup).
7. Feature engineering with word TF-IDF, character TF-IDF, and message statistics.
8. Chi-square feature selection to keep the strongest 75,000 features.
9. Three model initialisations: Complement Naive Bayes, logistic regression, and linear SVM.
10. Stratified training and hold-out evaluation.
11. Five-fold validation and model selection by spam F1 (accuracy breaks ties).
12. Predictions for new messages.
13. Saved reusable model (`artifacts/spam_classifier.joblib`).

## Run it

```powershell
python -m pip install -r requirements.txt
python train.py --data data/email.csv
```

To use the original source directly instead:

```powershell
python train.py --data "C:\Users\Admin\OneDrive\Desktop\MailSpam\email.csv"
```

## Outputs

After training, open these generated files:

- `outputs/eda_report.md` — cleaning summary and numeric data insights.
- `outputs/eda_overview.png` — class, message length, word count, and word-length analysis.
- `outputs/top_words.png` — high-frequency words for ham vs spam.
- `outputs/model_validation.csv` — five-fold validation comparison for all candidates.
- `outputs/confusion_matrix.png` — final untouched test-set results.
- `outputs/training_results.json` — machine-readable selection and evaluation results.
- `artifacts/spam_classifier.joblib` — final model trained on all clean data.

## Predict new samples

```powershell
python predict.py "Congratulations! You have won a free prize. Call now." "Can we meet at 6 pm?"
```

The pipeline picks the best validated candidate automatically. It evaluates once on a
stratified hold-out set, then retrains the selected pipeline on all clean data before
saving it for repeatable local use.

## Live Gmail labels (local laptop)

The project includes `gmail_watcher.py`: an opt-in Gmail API watcher that runs
on your laptop and applies **custom** labels to recent inbox mail. It never
deletes, sends, forwards, archives, or applies Gmail's built-in Spam label.

### One-time setup

1. In Google Cloud, enable the Gmail API and create an OAuth **Desktop app**
   client. Configure your own Gmail account as a test user and grant the
   `https://www.googleapis.com/auth/gmail.modify` scope.
2. Download the OAuth JSON file and save it as `secrets/credentials.json`.
   Do not commit, share, or upload it.
3. Install the dependencies and ensure the model has been trained:

```powershell
python -m pip install -r requirements.txt
python train.py --data data/email.csv
```

### Test safely, then label live email

The first command opens a browser for Google sign-in and makes **no** Gmail
changes. It prints predictions and writes only local metadata to
`outputs/gmail_predictions.jsonl`.

```powershell
python gmail_watcher.py --once
```

After reviewing the dry-run output, apply labels to messages processed from the
past seven days. Gmail will show `ML-Spam`, `ML-Ham`, and `ML-Checked` labels.
The active two-label policy gives a message `ML-Spam` only at a score of 0.80
or above; every other message receives `ML-Ham`. `ML-Checked` prevents a
message being processed again. `ML-Review` is a legacy label from an earlier,
more cautious policy and is no longer assigned.

```powershell
python gmail_watcher.py --apply-labels --once
```

If earlier model labels are wrong, remove only the project's labels (the email,
Gmail's own labels, and every other label remain unchanged):

```powershell
python gmail_watcher.py --reset-ml-labels --confirm-reset --max-messages 500
```

For continuous local monitoring every minute, leave this running while the
laptop is on (stop it with `Ctrl+C`):

```powershell
python gmail_watcher.py --apply-labels --interval 60
```

The displayed `spam_score` comes from the selected classifier's decision
margin. It is helpful for ranking messages but is not a calibrated probability.
Review `ML-Spam` predictions before building any automation that moves mail.

## Project files

| File or folder | Purpose |
|---|---|
| `data/email.csv` | Copy of the supplied training dataset. |
| `src/spam_pipeline.py` | Data cleaning, EDA, preprocessing, features, model training, evaluation, and inference helpers. |
| `train.py` | Runs the complete 13-step training workflow. |
| `predict.py` | Predicts ham/spam for manually supplied new text. |
| `gmail_watcher.py` | Reads Gmail subject/body through OAuth and applies only custom labels. |
| `artifacts/spam_classifier.joblib` | Saved selected model, generated by training. |
| `outputs/` | Generated charts, validation results, metrics, and local Gmail event log. |
| `secrets/credentials.json` | Google OAuth desktop-client file. Keep private. |
| `secrets/token.json` | Locally stored Gmail authorization token. Keep private. |

## Complete operating guide

### 1. Install packages

```powershell
cd C:\Users\Admin\OneDrive\Documents\ChatGPT\MailSpam
python -m pip install -r requirements.txt
```

Optional virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Train or retrain the model

```powershell
python train.py --data data\email.csv
```

This regenerates EDA reports, compares all three candidate models with five-fold
validation, evaluates the selected model on an untouched hold-out set, and
saves the final model. Open `outputs/model_validation.csv` and
`outputs/confusion_matrix.png` to inspect the result.

### 3. Test a message manually

```powershell
python predict.py "Congratulations, you won a free prize. Reply now." "Can we meet at 6 pm tomorrow?"
```

### 4. Configure Gmail OAuth once

1. In Google Cloud, enable **Gmail API** for your project.
2. Configure OAuth consent and add your Gmail account as a test user.
3. Create an OAuth client with type **Desktop app**.
4. Add the `https://www.googleapis.com/auth/gmail.modify` scope. It permits
   reading the message text and adding custom labels.
5. Save the downloaded JSON exactly as `secrets/credentials.json`.

The first watcher run opens Google sign-in. On approval, the project creates
`secrets/token.json`. If OAuth permissions change, delete only `token.json`
and run the watcher again. Never share either secrets file.

### 5. Test Gmail without changes

```powershell
python gmail_watcher.py --once
```

This reads up to 50 recent inbox messages from the last 7 days, prints each
local prediction, and writes a metadata-only local log to
`outputs/gmail_predictions.jsonl`. It does not change Gmail.

### 6. Apply Gmail labels

```powershell
python gmail_watcher.py --apply-labels --once
```

The current policy is:

| Model score | Gmail label |
|---|---|
| 0.80 or higher | `ML-Spam` |
| Below 0.80 | `ML-Ham` |

The score is a model ranking score, not a probability. A lower spam threshold
creates more `ML-Spam` labels but also increases false positives. The watcher
never deletes, sends, forwards, archives, or moves messages to Gmail's built-in
Spam folder.

### 7. Reset only project labels

If the labels are wrong, remove only `ML-Spam`, `ML-Ham`, `ML-Review`, and
`ML-Checked` from processed messages. The emails and Gmail system labels are
not altered.

```powershell
python gmail_watcher.py --reset-ml-labels --confirm-reset --max-messages 500
```

### 8. Process a larger range or monitor new mail

Process up to 500 messages from the latest 30 days:

```powershell
python gmail_watcher.py --apply-labels --once --max-messages 500 --lookback-days 30
```

Keep a local watcher running every 60 seconds while the laptop is on:

```powershell
python gmail_watcher.py --apply-labels --interval 60
```

Stop the watcher with `Ctrl+C`. The `--once` option runs one batch and stops.

## Gmail accuracy: current limitation and next step

This is a complete local machine-learning and Gmail-labeling project, but it
is not a validated production-grade Gmail spam filter yet. The SMS results do
not prove Gmail results. Your inbox already demonstrated this: job and account
emails received high spam scores despite being legitimate.

To improve Gmail accuracy, create a dataset of actual email examples you are
allowed to use, with the same two columns as the original data:

```csv
Category,Message
ham,"Subject: Application update. Body: Your application is under review."
spam,"Subject: You won a prize. Body: Pay a fee to claim immediately."
```

Use several hundred or more examples in each class, protect personal data, then
retrain and inspect the new metrics:

```powershell
python train.py --data data\gmail_training.csv
```

Only use continuous Gmail labeling after checking that the retrained model
performs well on email examples it did not train on.

## Troubleshooting

| Problem | What to do |
|---|---|
| `credentials.json` not found | Save the Google Desktop OAuth JSON as `secrets/credentials.json`. |
| Browser asks to sign in again | Keep `secrets/token.json`; delete it only after permission changes. |
| Only some messages are labeled | Increase `--max-messages` and `--lookback-days`. Defaults are 50 messages and 7 days. |
| New mail is not labeled | Use the continuous command; `--once` stops after one batch. |
| Labels look incorrect | Reset project labels, then retrain with labeled real email examples. |
| Gmail API permission error | Confirm Gmail API is enabled and your account is an OAuth test user in the same Cloud project. |

## Security and privacy

- `secrets/credentials.json` and `secrets/token.json` are private and ignored by Git.
- The Gmail watcher logs IDs, timestamps, labels, and scores locally; it does not log message bodies.
- Use a test inbox while improving thresholds and training data.
- Keep Gmail's own spam protection enabled. This project adds custom labels; it does not replace Gmail security.
