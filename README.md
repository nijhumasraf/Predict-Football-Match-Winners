# ⚽ Predict Football Match Winners

This project builds a **machine learning model** to predict the outcome of football matches using historical match data. The dataset includes various statistics such as team performance, previous results, and other key match attributes.

## 📂 Project Structure
- `football-predictor.ipynb` - Jupyter Notebook with data analysis, model training, and evaluation.
- `matches.csv` - Dataset containing historical match data.
- `models/` - Saved models for inference.
- `README.md` - Project documentation.

## 📊 Dataset: `matches.csv`
The dataset consists of football match statistics, including:
- **Actual Results** (`actual_x`, `actual_y`)
- **Predictions** (`prediction_x`, `prediction_y`)
- **Teams and Opponents** (`team_x`, `team_y`, `opponent_x`, `opponent_y`)
- **Match Outcome** (`result_x`, `result_y`)
- **Other Features** (`new_team_x`, `new_team_y`)

## 🛠️ Installation & Setup
1. **Clone the repository**:
   ```sh
   git clone https://github.com/nijhumasraf/Predict-Football-Match-Winners.git
   cd Predict-Football-Match-Winners
