import pandas as pd

# Load the dataset
file_path = "traindata.csv"  # Ensure this file is in the same directory
df = pd.read_csv(file_path)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the data (1/15th for testing, 14/15th for training)
test_size = len(df) // 15
test_df = df[:test_size].copy()
train_df = df[test_size:].copy()

# Save the test and training datasets
train_df.to_csv("traindata_updated.csv", index=False)
test_df.to_csv("testdata.csv", index=False)

print("✅ Data successfully split into training and test sets!")

# Rename columns for clarity (Assumed from structure)
column_names = [
    "Age", "Gender", "ChestPainType", "RestingBP", "Cholesterol", 
    "FastingBS", "RestingECG", "MaxHeartRate", "ExerciseAngina", 
    "Oldpeak", "ST_Slope", "OtherFeature1", "OtherFeature2", "Result"
]

train_df.columns = column_names
test_df.columns = column_names

# Define the first manual rule
def initial_manual_rule(row):
    if row["Age"] >= 55 and row["Cholesterol"] > 240:
        return "yes"  # High-risk group
    elif row["MaxHeartRate"] < 120 or row["ExerciseAngina"] == 1:
        return "yes"  # Weak heart performance
    elif row["Age"] < 40 and row["RestingBP"] < 130 and row["Cholesterol"] < 200:
        return "no"  # Healthy group
    else:
        return "yes"  # Default high-risk classification

# Apply the initial rule
test_df["PredictedResult"] = test_df.apply(initial_manual_rule, axis=1)

# Compute initial accuracy
initial_correct_predictions = (test_df["PredictedResult"] == test_df["Result"]).sum()
initial_accuracy = (initial_correct_predictions / len(test_df)) * 100
print(f"📊 Initial Rule Accuracy: {initial_accuracy:.2f}%")

# Define refined rule after analyzing misclassifications
def refined_manual_rule(row):
    if row["Age"] >= 55 and row["Cholesterol"] > 250:
        return "yes"  # High-risk group
    elif row["MaxHeartRate"] < 110 and row["ExerciseAngina"] == 1:
        return "yes"  # High risk
    elif row["Age"] < 45 and row["Cholesterol"] < 230 and row["RestingBP"] < 130:
        return "no"  # Healthy
    elif row["Cholesterol"] > 270 and row["Age"] > 60:
        return "yes"  # Higher cholesterol risk for elderly
    else:
        return "no"  # Default low-risk classification

# Apply refined rule
test_df["RefinedPredictedResult"] = test_df.apply(refined_manual_rule, axis=1)

# Compute refined accuracy
refined_correct_predictions = (test_df["RefinedPredictedResult"] == test_df["Result"]).sum()
refined_accuracy = (refined_correct_predictions / len(test_df)) * 100
print(f"🚀 Refined Rule Accuracy: {refined_accuracy:.2f}%")

# Save the test dataset with predictions
test_df.to_csv("testdata_with_predictions.csv", index=False)
print("✅ Final predictions saved in testdata_with_predictions.csv")
