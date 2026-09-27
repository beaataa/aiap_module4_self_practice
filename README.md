# Predicting Survival of Coronary Artery Disease Patients

Classification pipeline that predicts whether a coronary artery disease patient
survives, from the bloodwork and measurements in their medical record, so that
doctors can put preemptive treatment in front of the patients most at risk.

The best model reaches **0.887 ROC AUC** and 0.787 accuracy, against 0.850 for
logistic regression and 0.500 for predicting the majority class.

Those numbers are much lower than a first pass produces, and that is the point.
Trained the ordinary way, three of these models score a near-perfect 1.000 AUC.
That is not a result, it is a symptom: the 15000 rows in the file are **297
patients recorded 32 to 109 times each**, so a random split tests the model on
patients it has already memorised. Every split here is by patient instead. See
[How the main finding was found](#how-the-main-finding-was-found).

## Folder structure

```
.
├── data/
│   ├── classification_bonus_practice_data.csv   raw data, never modified
│   └── cleaned.csv                              output of the cleaning step
├── src/
│   ├── config.yaml                              all parameters live here
│   ├── config_schema.py                         validates config.yaml before use
│   ├── logging_setup.py                         one logger for every script
│   ├── data_preparation.py                      cleaning and feature engineering
│   ├── model_training.py                        split, models, evaluation, saves the model
│   └── predict.py                               scores new patients with the saved model
├── models/
│   ├── pipeline.joblib                          the fitted winning pipeline
│   └── model_metadata.json                      features, scores, threshold
├── tests/                                       69 pytest tests
│   ├── conftest.py                              fixtures, including hand-built messy rows
│   ├── test_config_schema.py                    11 tests
│   ├── test_data_preparation.py                 26 tests
│   ├── test_model_training.py                   19 tests
│   └── test_predict.py                          13 tests
├── results/
│   ├── model_comparison.csv                     every metric for every model
│   ├── stability_check.csv                      the ranking under 10 redraws of the folds
│   ├── leakage_check.csv                        what a row-level split would have claimed
│   ├── confusion_matrix.csv                     the chosen model on the test patients
│   ├── threshold_profile.csv                    precision and recall as the cut-off moves
│   ├── feature_importances.csv                  importances for the chosen model
│   ├── grid_search_results.csv                  every hyperparameter combination tried
│   └── best_params.csv                          chosen hyperparameters
├── logs/                                        timestamped run logs (gitignored)
├── eda.ipynb                                    exploration, and the reasoning behind the cleaning
├── requirements.txt
├── pytest.ini
└── run.sh
```

## Running it

```bash
pip install -r requirements.txt
./run.sh
```

`run.sh` runs the tests first, then cleaning, then training. If a test fails the
run stops before touching any data.

Or the steps separately, from the project root:

```bash
python -m pytest tests            # 69 tests, about 3 seconds
python src/data_preparation.py    # writes data/cleaned.csv
python src/model_training.py      # writes results/ and models/
```

### Scoring new patients

```bash
python src/predict.py data/new_patients.csv
python src/predict.py data/new_patients.csv --threshold 0.3 --out flagged.csv
```

The input csv must have the same columns as the raw file. `predict.py` imports
the cleaning functions from `data_preparation.py` rather than copying them, so
the two paths cannot drift apart, and it checks the input against the feature
list in `models/model_metadata.json` before scoring. If the file happens to
include a `Survive` column it also reports how well the model did.

One warning: scoring rows that were in training gives inflated numbers. Feeding
a slice of the original file back through `predict.py` reports about 0.95
accuracy, which is the same leak described below, not a real score. Use held-out
patients.

Training takes about two minutes, most of it in the stability and leakage
checks. Every parameter is in `src/config.yaml`, which is validated on load, so
a mistyped key is reported by name instead of failing partway through training.
Logs go to `logs/`.

## Pipeline

```
raw csv
   -> clean          fold duplicate label styles, fix sign errors, drop the decoys
   -> engineer       bmi, anaemia, high_bp, patient_key
   -> split 80/20    by patient, before anything is fitted
   -> preprocess     impute, one hot encode, scale (fitted on train folds only)
   -> train          5 models, 5 fold grouped cross validation, grid search
   -> evaluate       repeated cross validation chooses, test set scored once
   -> leakage check  the same models rescored on a random row split, for contrast
   -> save           the winning pipeline to models/, for predict.py
```

The split comes before any fitted transformation. Everything that learns a
value from the data sits inside a scikit-learn `Pipeline`, so cross validation
refits it on each fold and no test information reaches the training data.

This is why `data/cleaned.csv` still contains missing `Creatinine` values.
Imputing them during cleaning would have computed the median over the test rows
as well.

## How the main finding was found

`eda.ipynb` is written in the order the work actually happened, because the
wrong turn is what found the most important thing in the dataset. In short:

**The first pass looked normal.** Standardise the label styles, fix 430
negative ages, note the 499 missing `Creatinine`, drop `Favorite color` as an
obvious decoy. Nothing in any of that hints at a problem.

**So the first model was trained the ordinary way**, 80/20 split stratified on
`Survive` so both sides carry the same class balance. That is the textbook
approach, and it produced this:

| model | accuracy | precision | recall | F1 | AUC |
|---|---|---|---|---|---|
| Gradient Boosting | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| Random Forest | 0.999 | 1.000 | 0.998 | 0.999 | 1.000 |
| Decision Tree | 0.993 | 0.992 | 0.988 | 0.990 | 0.992 |
| Logistic Regression | 0.827 | 0.768 | 0.659 | 0.709 | 0.872 |
| Baseline (majority) | 0.679 | 0.000 | 0.000 | 0.000 | 0.500 |

**A flawless classifier is not a result, it is a symptom.** The shape of it is
the clue: logistic regression scores a believable 0.872, the trees score
essentially 1.000. So whatever is happening is something a tree can do and a
linear model cannot, and the obvious candidate is memorising individual
records. A tree can split until each leaf holds one patient; a linear model has
one coefficient per feature and nowhere to store 15000 people. That only wins
if the test set contains patients the model already saw.

### The 15000 rows are 297 patients

Going back to the data with that suspicion, the distinct-value counts give it
away, but only for the right columns:

| column | distinct values | range | suspicious? |
|---|---|---|---|
| `Age` | 46 | 40 to 95 | no, whole numbers in a narrow range |
| `Sodium` | 27 | 113 to 148 | no, same reason |
| `Platelets` | 176 | 25,100 to 850,000 | **yes** |
| `Creatine phosphokinase` | 208 | 23 to 7,861 | **yes** |

15000 real platelet counts spread over 25,100 to 850,000 should land on
thousands of different values. They land on 176.

Pasting the stable columns together and adding one at a time shows where the
rows collapse:

| fingerprint | groups | groups holding both outcomes |
|---|---|---|
| `Age` | 46 | 25 |
| `Age` + `Sodium` | 197 | 30 |
| `Age` + `Sodium` + `Platelets` | 294 | 3 |
| **the four above + `Creatine phosphokinase`** | **297** | **0** |

297 groups, each repeated 32 to 109 times, and not one contains both a survivor
and a non-survivor. Those four columns allow about 45 million combinations
(46 x 27 x 176 x 208); 15000 genuinely different patients dropped into 45
million slots would almost never collide.

Four more columns come out constant within those groups without having been
used to build them: `Creatinine`, `Gender`, `Smoke` and `Ejection Fraction`,
plus BMI once it is computed. That is the check that the fingerprint is picking
out patients rather than slicing arbitrarily.

### Confirming the leak

The original split put **100% of test rows** with a patient who also appears in
training. All 297 of 297 test patients had been seen. One test row next to its
copies:

| which set | Survive | Age | Sodium | Platelets | CPK | Creatinine | Height | Weight |
|---|---|---|---|---|---|---|---|---|
| **TEST** | 1 | 60 | 139 | 204000 | 47 | 0.7 | 187 | 114 |
| train | 1 | 60 | 139 | 204000 | 47 | 0.7 | 137 | 61 |
| train | 1 | 60 | 139 | 204000 | 47 | 0.7 | 153 | 77 |
| train | 1 | 60 | 139 | 204000 | 47 | 0.7 | 165 | 89 |

That patient appears 38 times in the training set. Identical bloodwork,
identical outcome, only the synthetic columns move. The tree was not predicting,
it was looking the answer up.

**The effective sample size is 297, not 15000.** Every split from here keeps a
patient wholly on one side, using `StratifiedGroupKFold`, which balances the
classes *and* respects the grouping. Stratifying and grouping are not
alternatives; this does both.

`src/model_training.py` reruns the comparison both ways and writes it to
`results/leakage_check.csv`, so the cost of getting the split wrong is measured
rather than asserted:

| model | AUC, random row split | AUC, patient split |
|---|---|---|
| Decision Tree | 0.9995 | 0.769 |
| Random Forest | 1.0000 | 0.868 |
| Gradient Boosting | 1.0000 | 0.848 |
| Logistic Regression | 0.876 | 0.851 |
| Baseline (majority) | 0.500 | 0.500 |

Logistic regression barely moves, exactly as predicted. The gap between 1.000
and 0.868 is the leak, measured.

## EDA findings

Two kinds of finding below. The label styles and the negative ages are basic
cleaning that any first pass catches. Everything else only became visible
*after* the patient grouping existed, because it depends on being able to ask
what changes within one person. Full working in `eda.ipynb`.

### Which columns belong to the patient, and which are made up

The patient groups are not just a fix for the split, they are the instrument
that found everything below. Once rows can be tied to a person, a new question
becomes available: does this column stay the same for one patient, or is it
redrawn on every copy? A real attribute cannot change. Anything that does was
generated.

| column | distinct values per patient | reading |
|---|---|---|
| `Gender`, `Age`, `Sodium`, `Creatinine`, `Platelets`, `Creatine phosphokinase` | 1.00 | real |
| `Diabetes` | 1.42 | real, two labels for one condition |
| `Ejection Fraction` | 1.81 | real, two label styles |
| `Smoke` | 1.88 | real, two label styles |
| `Favorite color` | 6.00 | redrawn every row |
| `Hemoglobin` | 29.3 | a flag hidden in a continuous column |
| `Weight` | 30.2 | half of BMI |
| `Height` | 34.1 | half of BMI |
| `Blood Pressure` | 36.2 | a flag hidden in a continuous column |

`Favorite color` is the only column that is noise all the way through: uniform
over six values, flat against survival (0.305 to 0.338), and redrawn inside
every patient. Keeping it is not expensive, 0.869 AUC against 0.870 over 10
redraws of the folds, so this is not a decision the score forces. It is dropped
because the evidence that it was made up is conclusive, and a model that never
sees it can never lean on it.

### `Hemoglobin` and `Blood Pressure` are yes/no flags

Each patient's hemoglobin readings sit in one of two disjoint bands, 9.0-11.9
or 12.0-17.5, and no patient has readings in both. Blood pressure does the same
at 120, and there is no reading of exactly 120 anywhere in the file. The spread
inside a band is noise, the band is the signal:

| | below the line | at or above |
|---|---|---|
| Hemoglobin 12.0 g/dL | 129 patients | 168 patients, 0 straddling |
| Blood Pressure 120 mmHg | 192 patients | 103 patients, 2 straddling |

Both become flags, `anaemia` and `high_bp`, and the continuous columns are
dropped. The two straddling patients are the two keys discussed under
Assumptions below.

### `bmi` is the strongest predictor, and it does not exist in the raw data

`Height` is uniform over 130-189 and redrawn on every row of a patient, so on
its own it is noise. `Weight` correlates 0.41 with survival, but what that
weight means depends entirely on the height it belongs to:

| weight | short patients | middle | tall |
|---|---|---|---|
| 19-46 kg | 0.107 | 0.043 | 0.005 |
| 46-60 | 0.244 | 0.161 | 0.088 |
| 60-73 | 0.515 | 0.233 | 0.159 |
| 73-92 | 0.883 | 0.392 | 0.230 |
| 92-141 | | 0.835 | 0.521 |

Reading across any row, the same weight means three different things.
`Weight / (Height/100)^2` removes it:

| BMI | short patients | middle | tall |
|---|---|---|---|
| 11.2-18.9 | 0.042 | 0.041 | 0.068 |
| 18.9-25.2 | 0.219 | 0.232 | 0.228 |
| 25.2-30.5 | 0.185 | 0.181 | 0.158 |
| 30.5-34.0 | 0.285 | 0.293 | 0.286 |
| 34.0-39.8 | 0.872 | 0.870 | 0.855 |

Flat across the three height groups. The decisive check is that BMI is constant
within a patient, 0.14 standard deviation against 7.73 across the dataset, even
though both columns it is built from vary freely. `Weight` was generated as BMI
times height squared. Both originals are dropped.

BMI correlates 0.53 with survival at patient level, twice the next strongest
feature, and accounts for 44% of what the trained forest does. Replacing it
with raw `Height` and `Weight` costs 0.064 AUC, from 0.870 to 0.806. This is
the largest single feature decision in the project.

### Two columns had their labels typed twice, and one condition was named twice

| column | the two styles | evidence they are the same |
|---|---|---|
| `Survive` | `0`/`1` on 14133 rows, `No`/`Yes` on 867 | survival 0.322 against 0.303 |
| `Smoke` | `Yes`/`No` on 14417 rows, `YES`/`NO` on 583 | 0.313 against 0.299, 0.324 against 0.356 |
| `Ejection Fraction` | `Low`/`Normal` on 14452, `L`/`N` on 548 | 0.339 against 0.346, 0.194 against 0.152 |
| `Diabetes` | `Diabetes` and `Pre-diabetes` | a diabetic patient's rows are split at random between the two |

`Diabetes` is the interesting one. A patient is either `Normal` on every row, or
has their rows split at random between `Diabetes` and `Pre-diabetes`. Two labels
for one binary condition, so `Pre-diabetes` folds into `Diabetes`.

### 430 ages are negative

Their absolute values sit on the same distribution as the positive ones: mean
60.0 against 60.9, same 40-95 range, same shape. Sign errors, recovered with
`abs()`. The pipeline raises if anything still falls outside 40-95 afterwards.

### Every missing `Creatinine` belongs to a patient who did not survive

499 rows have no creatinine reading. Counted per patient, 185 patients have at
least one gap and every one of them died; 96 patients survived and not one has a
gap. Under missing-at-random, 499 draws at a 32% survival rate would give about
160 survivors; there are zero. Fisher's exact test returns an odds ratio of
exactly 0 at `p = 1.3e-61`.

So a "was this value missing" indicator column is a copy of the target wearing
a feature's clothes. It is left out, and it costs almost nothing to leave out:
adding it moves grouped cross-validated AUC from 0.870 to 0.872, inside the
0.010 the number moves anyway when the folds are redrawn. Those patients are
already recognisable from their other readings. The 499 values get the median,
fitted on the training fold.

### What predicts survival, measured per patient

Per-row correlations would count a patient with 109 copies 109 times. These are
the 297 patients, one row each:

| feature | correlation with survival |
|---|---|
| `bmi` | 0.530 |
| `Creatinine` | 0.294 |
| `Age` | 0.255 |
| `Sodium` | -0.195 |
| `high_bp` | 0.086 |
| `Creatine phosphokinase` | 0.063 |
| `anaemia` | 0.062 |
| `Platelets` | -0.049 |

| category | groups |
|---|---|
| `Ejection Fraction` | Low 0.341 (261 patients), Normal 0.200 (35), High 0.000 (1) |
| `Gender` | Female 0.327, Male 0.321 |
| `Smoke` | No 0.328, Yes 0.312 |
| `Diabetes` | Normal 0.326, Diabetes 0.320 |

### Which of those differences survive a significance test

Averages on 297 patients can be luck, so each one is tested. Welch's t-test for
the numeric columns (the groups are 96 against 201, with unequal variances),
chi-square for the categoricals, and Benjamini-Hochberg across all twelve,
because testing twelve things at `p < 0.05` expects one false positive.

Every test runs at **patient level**. On 15000 rows the same tests would treat
50 copies of one person as 50 independent observations, inflate every sample
size fiftyfold, and report significance for almost everything.

| feature | test | effect size | raw p | adjusted p | verdict |
|---|---|---|---|---|---|
| `bmi` | Welch t | d = **1.33** | 1.7e-22 | <0.001 | **significant** |
| `Age` | Welch t | d = 0.56 | 4.6e-05 | <0.001 | **significant** |
| `Creatinine` | Welch t | d = 0.66 | 6.5e-05 | <0.001 | **significant** |
| `Sodium` | Welch t | d = -0.42 | 1.9e-03 | 0.006 | **significant** |
| `high_bp` | chi-square | V = 0.08 | 0.175 | 0.387 | no evidence |
| `Ejection Fraction` | chi-square | V = 0.11 | 0.194 | 0.387 | no evidence |
| `anaemia` | chi-square | V = 0.06 | 0.341 | 0.532 | no evidence |
| `Creatine phosphokinase` | Welch t | d = 0.13 | 0.368 | 0.532 | no evidence |
| `Platelets` | Welch t | d = -0.11 | 0.399 | 0.532 | no evidence |
| `Smoke` | chi-square | V = 0.01 | 0.888 | 1.000 | no evidence |
| `Gender` | chi-square | V = 0.00 | 1.000 | 1.000 | no evidence |
| `Diabetes` | chi-square | V = 0.00 | 1.000 | 1.000 | no evidence |

Four features clear the bar, the same four the correlations pointed at. BMI's
effect size of 1.33 is large by any convention, and its correlation confidence
interval is `[0.443, 0.607]`, nowhere near zero.

**Two of the non-results correct what the averages seemed to say.**

`Ejection Fraction` looked like the one clean categorical split, 0.341 against
0.200. Tested, it is `p = 0.19`, adjusted to 0.39. The group sizes explain it:
261 of 297 patients are `Low` and only 35 are `Normal`, so the whole comparison
rests on 35 people. The difference may well be real; this dataset cannot show
it.

`Creatine phosphokinase` fails at `p = 0.37` yet ranks **third** by importance
in the final model. Not a contradiction, but the t-test being the wrong
instrument: it compares two means, and CPK has no direction (0.23, 0.42, 0.31,
0.39, 0.27 across five bands), so the means match while the relationship
exists. A tree can use a shape a difference in means cannot see.

The lesson runs both ways: a p-value answers one narrow question, and it is not
the question the model is answering.

`Favorite color` has no patient-level value to test, since it is redrawn every
row. That helps the argument: tested on all 15000 rows, with fifty times more
power than it deserves, chi-square still gives `p = 0.219`.

All four weak categoricals are kept regardless. They are genuine clinical
variables rather than planted decoys, "no evidence of an effect" is not "no
effect", and dropping them moves AUC by less than the fold-to-fold spread in
either direction. The trained model ranks all four last, which is the honest
place for them.

### Three features do not move in a straight line with survival

| feature | what it does |
|---|---|
| `bmi` | nothing below 16 survives, almost everyone above 34 does, and the middle jumps around: 0.10 at 16-18, 0.06 at 18-20, 0.58 at 22-24, 0.14 at 24-26 |
| `Age` | dips through the fifties before climbing: 0.26 under 50, 0.18 at 50-58, 0.32 at 58-63, 0.29 at 63-70, 0.61 above 70 |
| `Creatine phosphokinase` | no direction at all: 0.23, 0.42, 0.31, 0.39, 0.27 across the five bands |

A linear model fits one coefficient per feature, so it gets all three of these
wrong. A tree can cut a feature into ranges and treat each range differently.
That is a reason to expect the tree models to do better here, and they do, by
0.037 AUC.

### Every column, and what happened to it

| column | decision | reason |
|---|---|---|
| `ID` | dropped | 885 codes shared by unrelated patients, one mangled to `8.46E+65` |
| `Survive` | target | `No`/`Yes` folded into `0`/`1` |
| `Gender` | kept | constant per patient, but 0.327 against 0.321 |
| `Smoke` | kept, labels standardised | `YES`/`NO` folded in. 0.328 against 0.312 |
| `Diabetes` | kept, labels merged | `Pre-diabetes` folded into `Diabetes`. 0.326 against 0.320 |
| `Age` | kept, signs fixed | 430 negative values. correlation 0.255 |
| `Ejection Fraction` | kept, labels standardised | `L`/`N` folded into `Low`/`Normal`. 0.341 / 0.200 / 0.000 |
| `Sodium` | kept | correlation -0.195 |
| `Creatinine` | kept | correlation 0.294. 499 gaps, median filled after the split, no indicator |
| `Platelets` | kept | correlation -0.049 |
| `Creatine phosphokinase` | kept | correlation 0.063, no direction. third by importance anyway |
| `Blood Pressure` | used, then dropped | two bands per patient, folded into `high_bp` |
| `Hemoglobin` | used, then dropped | two bands per patient, folded into `anaemia` |
| `Height` | used, then dropped | uniform noise alone, needed for `bmi` |
| `Weight` | used, then dropped | folded into `bmi` |
| `Favorite color` | dropped | uniform, flat, redrawn inside every patient |
| `bmi` | created | `Weight / (Height/100)^2`. strongest feature at 0.530 |
| `anaemia` | created | `Hemoglobin < 12` |
| `high_bp` | created | `Blood Pressure >= 120` |
| `patient_key` | created, used, then dropped | splits the data by patient, predicts nothing |

## Assumptions

Each of these is a judgement call, listed so it can be challenged:

1. **Rows sharing `Age`, `Sodium`, `Platelets` and `Creatine phosphokinase` are
   the same patient.** The evidence is that the outcome and five unused columns
   are constant inside all 297 groups. The alternative reading, that these are
   distinct patients who match on four lab values and always share an outcome,
   is not credible. Everything downstream rests on this.
2. **Two of the 297 keys are two patients each.** Both carry
   `Platelets = 263358.03`, a filled-in average shared by 1245 rows, and both
   are about twice the size of anything else in the file at 109 and 96 rows.
   They are also the only keys where `Diabetes` or `Blood Pressure` is
   internally inconsistent. Treating two patients as one makes the grouping
   conservative rather than leaky, so they are left merged.
3. **Negative ages are sign errors** (430 rows), not a different kind of
   record. The absolute values match the positive distribution. The pipeline
   raises if any age outside 40-95 survives cleaning.
4. **`No`/`Yes`, `NO`/`YES` and `L`/`N` are alternative spellings**, not
   distinct categories. Each pair scores within noise of the other.
5. **`Pre-diabetes` and `Diabetes` are one condition.** No patient is ever
   `Normal` on some rows and diabetic on others, and a diabetic patient's rows
   are split at random between the two labels.
6. **`Favorite color` carries no signal** and is dropped. Uniform, flat, and
   redrawn inside every patient.
7. **`Height` and `Weight` are one feature.** BMI is constant within a patient
   while both inputs vary, so weight was derived from BMI and height rather
   than the other way round.
8. **`Hemoglobin` below 12 and `Blood Pressure` at 120 or above are the real
   variables.** Each column is two disjoint bands per patient with nothing in
   between, so the within-band spread is noise.
9. **Missing `Creatinine` must not be used as a feature.** All 499 gaps sit in
   patients who did not survive, so the missingness was generated from the
   outcome. A missing indicator would be target leakage rather than a signal a
   doctor could act on, and it is switched off in `config.yaml`.
10. **`Survive = 1` means the patient survived**, as the brief states. Low
    ejection fraction, high creatinine, low sodium and high age all point at
    class 1 here, which is clinically backwards and suggests the column may be
    a death indicator renamed. Nothing in the file settles it, so the brief is
    taken at its word. If it is reversed, every model and every number below is
    unchanged and only the word "survived" swaps for "died".

## Features

| feature | type | treatment |
|---|---|---|
| `Age` | numeric | scaled |
| `Sodium` | numeric | scaled |
| `Creatinine` | numeric | median imputed, scaled |
| `Platelets` | numeric | scaled |
| `Creatine phosphokinase` | numeric | scaled |
| `bmi` | numeric, engineered | `Weight / (Height/100)^2`, scaled |
| `anaemia` | binary, engineered | `Hemoglobin < 12` |
| `high_bp` | binary, engineered | `Blood Pressure >= 120` |
| `Gender` | categorical | one hot |
| `Smoke` | categorical | one hot |
| `Diabetes` | categorical | one hot |
| `Ejection Fraction` | categorical | one hot |

`patient_key` is used to split the data and is then dropped. It identifies a
patient, so handing it to the model would be handing it the answer.

Scaling only matters to logistic regression, but applying it to every model
keeps a single preprocessor and makes the comparison fair.

## Models

Four models plus a reference line, all through the same preprocessing pipeline:

| model | what it does |
|---|---|
| Baseline (majority) | predicts "did not survive" for everyone. Not a model, just the score everything else has to beat |
| Logistic Regression | fits one coefficient per feature and pushes the sum through an S curve. Coefficients can be read directly |
| Decision Tree | cuts each feature into ranges and predicts a rate inside each range |
| Random Forest | builds many trees on random subsets of the data and averages their votes |
| Gradient Boosting | builds trees one after another, each one correcting the errors of the last |

Every grid in `config.yaml` has exactly 12 combinations, so no model wins by
having been searched harder than another. All 48 are in
`results/grid_search_results.csv`.

### How the models are compared

Three separate numbers, because on 297 patients a single number does not settle
anything:

- **5-fold grouped cross validation** on the 238 training patients, tuning on
  ROC AUC. Grouped, so a patient's copies stay in one fold.
- **The same cross validation redrawn 10 times** with different fold
  assignments. One fold is about 48 patients, and 48 patients move an AUC by
  more than the gap between the top two models. This is the number the
  comparison is ranked on, and it is what makes the ranking trustworthy rather
  than lucky.
- **The 59 held-out test patients**, scored once at the end, after every
  decision had been made.

ROC AUC is the headline metric because it does not depend on where the decision
threshold sits, and the threshold here is a clinical choice rather than a
modelling one. Accuracy, precision, recall and F1 are reported alongside it in
`results/model_comparison.csv`, and `results/threshold_profile.csv` shows what
moving the cut-off does.

## Results

Cross validation is on the 238 training patients, test is the 59 held-out
patients. The AUC column is the mean over 10 redraws of the folds; the rest of
the cross-validation columns come from the single seeded draw.

| model | CV AUC | +- | CV F1 | CV acc | | Test AUC | Test F1 | Test precision | Test recall | Test acc |
|---|---|---|---|---|:-:|---|---|---|---|---|
| **Random Forest** | **0.887** | 0.009 | 0.746 | 0.850 | │ | 0.814 | 0.642 | 0.692 | 0.598 | 0.787 |
| Gradient Boosting | 0.885 | 0.013 | 0.758 | 0.845 | │ | 0.799 | 0.612 | 0.671 | 0.563 | 0.773 |
| Logistic Regression | 0.850 | 0.010 | 0.695 | 0.771 | │ | 0.852 | 0.635 | 0.587 | 0.691 | 0.746 |
| Decision Tree | 0.800 | 0.023 | 0.693 | 0.808 | │ | 0.783 | 0.667 | 0.698 | 0.639 | 0.797 |
| Baseline (majority) | 0.500 | 0.000 | 0.000 | 0.678 | │ | 0.500 | 0.000 | 0.000 | 0.000 | 0.681 |

Chosen hyperparameters: Logistic Regression `C=0.1, class_weight=balanced`,
Decision Tree `max_depth=5, min_samples_leaf=100`, Random Forest `max_depth=10,
min_samples_leaf=20, max_features=sqrt, n_estimators=300`, Gradient Boosting
`learning_rate=0.2, max_depth=6, max_iter=300`.

### Reading the results

The baseline ignores every column and calls everyone a non-survivor. That is
right 68.1% of the time, which is exactly why accuracy alone is the wrong way
to read this table: three of the four models sit within 5 points of a model
that has learned nothing. AUC is not fooled the same way. The baseline scores
0.500, that is the bar, and every model clears it.

Logistic regression gets most of the way there, 0.850. The ensembles add 0.037
on top. That gap is the three non-linear features from the EDA: once a model can
treat different ranges of `bmi`, `Age` and `Creatine phosphokinase`
differently, it picks up something a single coefficient per feature cannot
reach.

A single Decision Tree is the worst of the four at 0.800, which is the opposite
of what the same comparison shows on well-populated data. With 238 training
patients a lone tree has too little to cut on and its folds swing by 0.023, the
widest spread in the table. Averaging 300 of them fixes exactly that.

The test column disagrees with the cross-validation column, and the
disagreement is the most useful thing in it. Logistic regression is fourth on
cross validation and first on test. **59 patients is not enough to rank
models.** Swapping which 59 are held out moves an AUC by more than the entire
spread between the top three. This is why the comparison is ranked on the
repeated cross validation, which scores all 238 training patients ten times
over, and why the test set is reported as a sanity check rather than used to
choose.

### Model comparison, and why Random Forest

Across the 10 redraws in `results/stability_check.csv`:

| model | mean AUC | std | worst draw | best draw | ranked first |
|---|---|---|---|---|---|
| Random Forest | 0.887 | 0.009 | 0.875 | 0.902 | 5 of 10 |
| Gradient Boosting | 0.885 | 0.013 | 0.861 | 0.907 | 5 of 10 |
| Logistic Regression | 0.850 | 0.010 | 0.831 | 0.863 | 0 of 10 |
| Decision Tree | 0.800 | 0.023 | 0.755 | 0.835 | 0 of 10 |

This is the comparison the brief asks for, and it answers three questions at
once:

**Random Forest against Gradient Boosting: a tie.** 0.887 against 0.885 is
noise, and the 5-5 split across the redraws says so directly. Nothing separates
them on accuracy. What separates them is steadiness: the forest's spread across
the 10 redraws is 0.009 against 0.013, and its worst draw is 0.875 against
0.861. It is slightly less sensitive to which patients it happens to be shown,
which matters when there are only 297 of them. It is also about three times
faster to fit, which is no concern at this size but breaks a tie when nothing
else does.

**Both beat Logistic Regression, and that gap is real.** 0.037 AUC, with the
worst forest draw (0.875) still above the best logistic draw (0.863). The two
never overlap across 10 redraws. The EDA predicted this: `bmi`, `Age` and
`Creatine phosphokinase` all fail to move in one direction, and a linear model
has one coefficient per feature to spend.

**The single Decision Tree is genuinely last.** 0.800, and the widest spread of
the four.

**Random Forest.** Best mean AUC, tightest spread, cheapest of the two
ensembles.

If explainability mattered more than accuracy, the honest alternative is
Logistic Regression rather than the Decision Tree. It costs 0.037 AUC, and in
return every prediction is a weighted sum a clinician can read off: BMI +1.59,
creatinine +1.03, low ejection fraction +0.64, sodium -0.31. The single tree is
0.087 AUC worse and swings most between folds, so the usual explainability
argument for it does not survive the small sample here. Random Forest is chosen
because the brief asks for the best estimate of survival, but Logistic
Regression is the one to reach for if the model has to justify itself.

### What the chosen model gets right and wrong

Random Forest on the 59 test patients (3001 rows), at the default 0.5 cut-off:

| | actually did not survive | actually survived |
|---|---|---|
| **predicted did not survive** | 1791 | 384 |
| **predicted survived** | 254 | 572 |

Accuracy 0.787, precision 0.692, recall 0.598. The model misses about 40% of
survivors at this threshold, which is the imbalance showing: two thirds of the
data is one class, and an untuned cut-off drifts toward it.

The cut-off is a clinical decision, not a modelling one, so
`results/threshold_profile.csv` prints the trade instead of leaving it at 0.5:

| threshold | flagged | precision | recall | F1 | accuracy |
|---|---|---|---|---|---|
| 0.2 | 1274 | 0.602 | 0.802 | 0.688 | 0.768 |
| 0.3 | 925 | 0.681 | 0.659 | 0.670 | 0.793 |
| 0.4 | 866 | 0.661 | 0.598 | 0.628 | 0.774 |
| 0.5 | 826 | 0.692 | 0.598 | 0.642 | 0.787 |
| 0.6 | 686 | 0.678 | 0.486 | 0.566 | 0.763 |
| 0.7 | 494 | 0.553 | 0.286 | 0.377 | 0.699 |
| 0.8 | 234 | 0.756 | 0.185 | 0.298 | 0.721 |

Dropping the cut-off to 0.2 lifts recall from 0.598 to 0.802 and costs 0.09 of
precision. For a tool whose purpose is deciding which patients get preemptive
treatment, missing a patient who needed it is the more expensive error, so the
lower threshold is the better default. 0.3 is the compromise, and is the best
accuracy in the table as well.

### What drives the prediction

A trained Random Forest reports how much each feature contributed. The numbers
add to 1, so 0.440 means that feature accounts for about 44% of what the model
does. One hot columns are added back together here, so each row is one original
column; the saved `results/feature_importances.csv` lists them unmerged.

| feature | importance |
|---|---|
| `bmi` | 0.440 |
| `Creatinine` | 0.165 |
| `Creatine phosphokinase` | 0.098 |
| `Platelets` | 0.075 |
| `Sodium` | 0.071 |
| `Age` | 0.068 |
| `Smoke` | 0.019 |
| `Diabetes` | 0.016 |
| `Gender` | 0.015 |
| `high_bp` | 0.013 |
| `anaemia` | 0.011 |
| `Ejection Fraction` | 0.009 |

`bmi` at 0.440 is the headline, and it had to be built first. `Gender`, `Smoke`
and `Diabetes` sit near the bottom, as the EDA predicted.

Two rows do not match the EDA and both have the same explanation.
`Creatine phosphokinase` ranks third on importance despite a patient-level
correlation of 0.063. It has 208 distinct values, which gives a tree a great
many places to split, and correlation only measures the straight-line part of a
relationship it does not have. The importance is real, the correlation was the
wrong instrument.

`Ejection Fraction` ranks last despite being the clearest categorical split in
the data, 0.341 against 0.200. That is because 88% of patients are `Low`: the
column separates a group the model rarely has to decide about. A split that
isolates 35 patients out of 297 cannot carry much of a forest's total impurity
reduction, however clean the separation inside it is.

## Engineering notes

The parts that make this runnable by someone other than its author.

**Config validation.** `src/config_schema.py` checks `config.yaml` on load and
reports every problem at once, not just the first:

```
6 problem(s) in config.yaml:
  - missing key: features.bmi_from
  - preprocessing.numeric_impute_strategy is 'avarage', expected one of [...]
  - evaluation.cv_folds is 1, needs to be at least 2
  - evaluation.ranking_metric is 'test_rmse', which is not a column the comparison produces
  - split.group_column is 'patient_id' but patient_key.column is 'patient_key'
  - patient_key.column 'patient_key' is missing from drop_for_modelling
```

The last two are the ones worth having. A mismatch between `split.group_column`
and `patient_key.column` would silently disable the grouped split, and leaving
`patient_key` out of `drop_for_modelling` would hand the model a per-patient
identifier. Both would produce plausible-looking output and a broken result.

**Logging.** `src/logging_setup.py` gives both scripts one timestamped logger,
writing to the console and to `logs/`. No `print()` calls remain in `src/`.

**Saved artifacts.** Training writes `models/pipeline.joblib` plus
`models/model_metadata.json`, which records the feature list and order, the
estimator type, the metric it was chosen by, its scores, and the default
threshold. The pipeline alone is not enough to reuse, because whoever loads it
has to know which columns to pass.

**Tests.** 69 tests, about 3 seconds, run first by `run.sh`.

| file | tests | covers |
|---|---|---|
| `test_config_schema.py` | 11 | the shipped config passes; each kind of breakage is caught |
| `test_data_preparation.py` | 26 | every cleaning rule, tested against the case that motivated it |
| `test_model_training.py` | 19 | the split, preprocessing, thresholds |
| `test_predict.py` | 13 | the inference round trip |

The fixtures in `conftest.py` are four hand-written rows rather than a sample,
each covering a specific mess: both target encodings, both smoke cases, both
ejection fraction styles, a negative age, a missing creatinine, `Pre-diabetes`,
and hemoglobin at exactly 11.9 and 12.0 to pin the anaemia boundary.

The most important test is this one:

```python
def test_no_patient_appears_on_both_sides(self, split):
    """the invariant the whole project rests on."""
    ...
    assert test_patients & train_patients == set()
```

Every score in this project depends on that holding. It is asserted rather than
assumed. Alongside it, `test_data_preparation.py` checks that all 297 patients
carry exactly one outcome, that BMI really is constant within a patient, and that
`data/cleaned.csv` still has its 499 missing creatinine values, since imputing
them during cleaning would compute the median over the test rows.

## Limitations

- **297 patients is a small dataset wearing a large one's clothes.** Every
  confidence interval in this project is the width you would expect from 297
  records, not 15000. The 0.002 AUC between Random Forest and Gradient Boosting
  is not a real difference, and the repeated cross validation is there to say
  so rather than to hide it.
- **The test metrics weight patients by how many copies they have.** A test
  patient with 109 rows counts twice as much as one with 50. Since every row of
  a patient carries the same outcome and nearly the same features, this shifts
  the test numbers a little without changing the ranking. Scoring one row per
  patient would be cleaner and was not done.
- **The importance numbers flatter columns with many distinct values.** Of the
  ways to measure importance, this one counts how often a column gets used to
  split. `Creatine phosphokinase` has 208 possible values and `Gender` has 2,
  so the first picks up a higher score partly for that reason alone. A fairer
  check is permutation importance, which shuffles one column and measures the
  damage. It was not run here.
- **The threshold was chosen by looking at the test set.** The table above is
  computed on the held-out patients, so the recommendation to use 0.3 or 0.2
  has seen the data it is recommended on. A stricter version would pick the
  threshold on a validation fold.
- **`Creatinine` is imputed with a median even though the missingness is
  informative.** All 499 gaps are in non-survivors. Refusing to use that is the
  right call, but it means those rows carry a value the patient never had.
- **The model shows what goes together, not what causes what.** High BMI and
  survival appear together in this data, but that is not evidence that changing
  a patient's weight would change their outcome, and the direction is
  clinically backwards to begin with (see Assumption 10). This is a tool for
  ranking which patients to look at, not for deciding what to do to them.
