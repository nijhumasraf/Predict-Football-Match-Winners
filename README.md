# Predict Football Match Winners using Machine Learning

This project is designed to **predict football match outcomes** using **machine learning** based on historical match data. As a **football freak** who plays and watches the game daily, I wanted to build a model that could analyze past matches and provide predictions. The goal is to help **football enthusiasts** and **bettors** make informed decisions using data-driven insights.

---

## Dataset

The dataset (`matches.csv`) consists of **English Premier League (EPL)** match data, including key match statistics. The dataset originally contained **1,520** matches, but after filtering out relegated teams, it was reduced to **1,389** matches.

### **Columns in the Dataset**
- **Match Info**: `date`, `time`, `season`, `round`, `day`, `venue`, `result`
- **Performance Stats**: `gf` (goals for), `ga` (goals against), `sh` (shots), `sot` (shots on target), `dist` (distance of shots), `fk` (free kicks), `pk` (penalty kicks), `pkatt` (penalty kick attempts)
- **Additional Factors**: `opponent`, `xg` (expected goals), `xga` (expected goals against), `poss` (possession %), `attendance`, `formation`, `referee`
- **Categorical Features**: `venue_code`, `opp_code`, `hour`, `day_code`, `target` (win = 1, loss/draw = 0)

---

## Data Preprocessing

Since some teams **get relegated** each season, they are removed from the dataset. I also cleaned the data by:
✅ Removing **duplicates**  
✅ Converting categorical features into **integer values** for model training  
✅ Filtering matches to **only include EPL games**  

To improve accuracy, I introduced **rolling averages** for key performance metrics:

```python
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]
```

This calculates the **rolling averages for the past 3 matches**, providing better predictive power.

---

## Machine Learning Model Used

I used a **Random Forest Classifier** as the main model. However, I plan to explore **other models** in the future for better accuracy.

### **Model Performance**
| Model                  | Accuracy | Precision |
|------------------------|----------|-----------|
| Initial Random Forest  | 61.2%    | 47.4%     |
| With Rolling Averages  | **65.7%** | **67.5%** |

By incorporating **rolling averages of past 3 matches**, the accuracy improved from **61.2% to 65.7%**, and precision increased from **47.4% to 67.5%**.

### **Train-Test Split**
- **Train Data**: Matches before **January 1, 2022**
- **Test Data**: Matches after **January 1, 2022**

```python
train = data[data["date"] < '2022-01-01']
test = data[data["date"] > '2022-01-01']
```

---

## Features Used for Prediction
- **Date** (match date)
- **Venue** (home/away)
- **Opponent**
- **Match Time** (hour of the match)
- **Day of the Week**
- **Result** (`1 = Win`, `0 = Lose/Draw`)

---

## Future Improvements
🔹 **Use more seasons**: Adding additional years of data will improve model accuracy.  
🔹 **Incorporate more features**: Using advanced stats like player performance, injuries, and weather conditions.  
🔹 **Try different ML models**: Exploring **XGBoost, Neural Networks, or Deep Learning** to identify non-linear trends.  
🔹 **Expand beyond EPL**: Including data from **other leagues** for broader insights.

---

## Challenges Faced
1. **Data Cleaning**: Converting categorical data into numerical values was challenging.  
2. **Feature Selection**: Finding the **most relevant** columns for predicting match outcomes required experimentation.  
3. **Conflicting Predictions**: If **both teams were predicted to win**, I had to create logic for handling conflicts:

```python
merged[(merged["prediction_x"] == 1) & (merged["prediction_y"] == 1)]["actual_x"].value_counts()
```
This helped determine the true winner when both teams were expected to win.

---
