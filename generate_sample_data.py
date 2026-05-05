"""Generate a realistic gym members exercise dataset."""
import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 500

age = np.random.randint(18, 65, N)
gender = np.random.choice(["Male", "Female"], N, p=[0.55, 0.45])
weight = np.where(gender == "Male",
                  np.random.normal(78, 12, N),
                  np.random.normal(62, 10, N)).round(1)
height = np.where(gender == "Male",
                  np.random.normal(175, 8, N),
                  np.random.normal(163, 7, N)).round(1)
bmi = (weight / (height / 100) ** 2).round(1)

workout_type = np.random.choice(
    ["Cardio", "Strength", "HIIT", "Yoga", "Stretching"], N,
    p=[0.30, 0.25, 0.20, 0.15, 0.10])

session_duration = np.clip(np.random.normal(1.2, 0.4, N), 0.3, 3.0).round(2)

# Calories depend on workout type and duration
base_cal = {"Cardio": 350, "Strength": 280, "HIIT": 420, "Yoga": 180, "Stretching": 120}
calories = np.array([
    base_cal[wt] * dur + np.random.normal(0, 40)
    for wt, dur in zip(workout_type, session_duration)
]).round(0).astype(int)
calories = np.clip(calories, 50, 900)

avg_heart_rate = np.clip(np.random.normal(135, 18, N), 80, 200).round(0)
max_heart_rate = (avg_heart_rate + np.random.randint(10, 35, N)).clip(100, 220)

fat_pct = np.where(gender == "Male",
                   np.random.normal(20, 6, N),
                   np.random.normal(28, 7, N)).round(1)
fat_pct = np.clip(fat_pct, 5, 45)

water_intake = np.clip(np.random.normal(2.2, 0.7, N), 0.5, 5.0).round(1)
workout_freq = np.random.choice([2, 3, 4, 5, 6, 7], N, p=[0.10, 0.20, 0.25, 0.20, 0.15, 0.10])
experience = np.where(workout_freq >= 5,
                      np.random.choice(["Intermediate", "Advanced"], N),
                      np.random.choice(["Beginner", "Intermediate"], N))

# Introduce ~3% missing values
for col_arr in [fat_pct, water_intake, avg_heart_rate]:
    mask = np.random.random(N) < 0.03
    col_arr[mask] = np.nan

df = pd.DataFrame({
    "Age": age,
    "Gender": gender,
    "Weight_kg": weight,
    "Height_cm": height,
    "BMI": bmi,
    "Workout_Type": workout_type,
    "Session_Duration_hrs": session_duration,
    "Calories_Burned": calories,
    "Avg_Heart_Rate": avg_heart_rate,
    "Max_Heart_Rate": max_heart_rate,
    "Fat_Percentage": fat_pct,
    "Water_Intake_liters": water_intake,
    "Workout_Frequency_days_week": workout_freq,
    "Experience_Level": experience,
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/gym_members.csv", index=False)
print(f"Generated {len(df)} rows, {len(df.columns)} columns")
print(df.head())
print("\nMissing values:\n", df.isnull().sum())
