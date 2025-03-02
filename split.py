import pandas as pd

# Load the data
file_path = "traindata.csv"  # Make sure the file is in the same directory as the script
df = pd.read_csv(file_path)

# Shuffle the data for randomness
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the data (1/15th for testing, 14/15th for training)
test_size = len(df) // 15
test_df = df[:test_size]
train_df = df[test_size:]

# Save the test and training data
test_file_path = "testdata.csv"
train_file_path = "traindata_updated.csv"

test_df.to_csv(test_file_path, index=False)
train_df.to_csv(train_file_path, index=False)

print(f"Test data saved as {test_file_path}")
print(f"Updated training data saved as {train_file_path}")
