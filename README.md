
NFL Play Type Prediction Using LSTM & Pre‑Snap Box Count Dynamics
IBM Machine Learning Professional Certificate — Deep Learning Final Project
Author: Eric Maurer
Dataset: NFL Big Data Bowl 2025 (Next Gen Stats Tracking, Weeks 1–9)

🚀 Project Overview
This project predicts whether an NFL offensive play will be a run or a pass using only pre‑snap information.
The key innovation is a set of box count time‑series features that track how many defenders are near the line of scrimmage at multiple time windows before the snap (T‑20s, T‑10s, T‑5s, T‑2s).

These features capture defensive loading and late adjustments, which often signal whether the offense will stick with the called play or audible out.

The model uses Long Short‑Term Memory (LSTM) networks to learn sequential patterns across plays within a game.

🎯 Research Questions
Can pre‑snap situational features predict run vs pass with meaningful accuracy?

Do late defensive box movements influence offensive play‑calling?

Which LSTM architectural choices (depth, dropout, learning rate) improve performance?

🧠 Model Architectures Compared
Model	Architecture	Purpose
Model 1	LSTM(64)	Baseline
Model 2	LSTM(128 → 64) + Dropout	Depth + regularization
Model 3	LSTM(32), lr=0.0005, batch=64	Hyperparameter exploration


📂 Repository Structure
Code
nfl-play-type-prediction/
│
├── README.md
├── nfl_play_prediction_lstm.ipynb
├── requirements.txt
├── .gitignore
│
├── figures/
│   ├── eda_class_balance.png
│   ├── eda_formation_pass_rate.png
│   ├── eda_box_counts.png
│   ├── eda_box_delta_vs_play_call.png
│   ├── eda_feature_distributions.png
│   ├── eda_sequence_lengths.png
│   ├── learning_curves.png
│   ├── f1_comparison.png
│   └── confusion_matrices.png
│
└── src/ (optional)
    ├── feature_engineering.py
    ├── sequence_builder.py
    └── model_definitions.py
📊 Feature Engineering
Situational Features
Down, distance, yardline

Score differential

Expected points

Play clock

Win probability

Formation & Coverage
Offense formation (one‑hot)

Receiver alignment (one‑hot)

PFF coverage & man/zone (one‑hot)

Motion Features
Aggregated from player_play.csv:

Any motion

Motion at snap

Any shift

⭐ Box Count Time‑Series (Novel Feature)
Defenders in the box (within 3 yards of LOS, within 8 yards depth) at:

T‑20s

T‑10s

T‑5s

T‑2s

Plus deltas:

box_delta_20_to_2

box_delta_10_to_2

box_delta_5_to_2

box_delta_10_to_5

These capture late defensive loading, a key predictor of play type.

🔍 Exploratory Data Analysis (Highlights)
Class Balance
Runs: 60.4%

Passes: 39.6%

<img width="1784" height="583" alt="image" src="https://github.com/user-attachments/assets/d71a6410-f470-4801-b6c0-53001e565088" />


Pass Rate by Down
1st: 49.7%

2nd: 62.0%

3rd: 80.0%

4th: 69.4%

Formation Tendencies
EMPTY: 96.5% pass

SHOTGUN: 74.5% pass

Under‑center formations: run‑heavy

⭐ Box Count Insights
Runs consistently show higher box counts

Late defensive loading (+2 or more defenders) → higher pass rate

Offenses tend to audible away from runs when the box fills late


<img width="2385" height="1217" alt="image" src="https://github.com/user-attachments/assets/1b77c922-6491-43dc-94d9-99d8069c6b91" />

<img width="1934" height="733" alt="image" src="https://github.com/user-attachments/assets/71ce2232-e91c-4fd4-a642-fb8bcd18141e" />

🧩 Sequence Construction
Each game = one sequence

Each play = one timestep

Features = full engineered vector

Temporal split:

Train: Weeks 1–7

Test: Weeks 8–9

Sequence lengths average 118 plays per game.

<img width="1483" height="581" alt="image" src="https://github.com/user-attachments/assets/89945875-4660-4c39-8127-3865cb1736c1" />

🏋️ Model Training & Learning Curves
Model 1 shows stable training and validation curves

Model 2 overfits

Model 3 underfits due to low learning rate

<img width="2084" height="763" alt="image" src="https://github.com/user-attachments/assets/5b30c145-e8e1-435b-9058-03ec8d375173" />

🧪 Results
Accuracy (Test Set)
Model	Accuracy
Model 1	67.5%
Model 2	62.9%
Model 3	60.0%


F1 Scores
Model	F1 Run	F1 Pass	F1 Macro
Model 1	0.41	0.78	0.59
Model 2	0.24	0.75	0.50
Model 3	0.03	0.75	0.39

<img width="1634" height="731" alt="image" src="https://github.com/user-attachments/assets/c9ad08e0-ea9d-401b-8b91-937c13f045c8" />

Model 1 is the clear winner.

Confusion Matrix Insights
Model 1 — Best Balanced Performance
Run correctly predicted: 28%

Pass correctly predicted: 94.6%

Most runs misclassified as passes (common in pre‑snap prediction)

Model 2 — Overfits
Run detection drops to 14.8%

Model 3 — Collapses on Run
Run detection: 1.6%

Predicts “pass” almost always

<img width="2234" height="618" alt="image" src="https://github.com/user-attachments/assets/55cb0c3d-1cee-4bff-9416-31770ba88c5a" />


🧠 Discussion
What Worked
Box count features significantly improved model understanding of defensive intent

LSTM captured sequential tendencies across games

Model 1 struck the best balance between complexity and generalization

What Didn’t
Deeper LSTM (Model 2) overfit

Lower learning rate (Model 3) underfit

Run plays remain harder to predict due to subtle pre‑snap cues

🏁 Conclusion
This project demonstrates that pre‑snap information alone can predict run vs pass with ~67% accuracy, and that defensive box dynamics are a meaningful signal.

Key takeaways:

Late defensive loading strongly correlates with offensive play choice

LSTM models can learn game‑level sequential patterns

A simple LSTM(64) outperforms deeper or slower‑learning variants

Predicting runs remains challenging due to weaker pre‑snap signatures

🔧 How to Run
Clone the repo

Install dependencies:

Code
pip install -r requirements.txt
Open the notebook:

Code
jupyter notebook nfl_play_prediction_lstm.ipynb
Add your local Big Data Bowl dataset (not included due to size & licensing)

📌 Future Work
Incorporate post‑snap frames (first 0.5 seconds)

Add offensive line splits and receiver splits

Try Transformer‑based sequence models

Explore team‑specific tendencies

Due to archived data restrictions, the data is on the kaggle site... https://www.kaggle.com/code/ericmau/nfl-play-type-prediction-lstm
