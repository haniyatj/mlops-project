import pandas as pd

# Load the CSV with correct delimiter handling
df = pd.read_csv("../data/AAPL_daily.csv")

# Strip column names of leading/trailing spaces
df.columns = df.columns.str.strip()

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Convert numeric columns to float (remove commas if needed)
numeric_cols = ['open', 'high', 'low', 'close', 'volume']
for col in numeric_cols:
    df[col] = df[col].astype(str).str.replace(',', '')  # Remove any commas
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to float

# Drop any rows with missing or invalid data
df.dropna(inplace=True)

# Remove duplicates (if any)
df.drop_duplicates(inplace=True)

# Sort by date
df.sort_values('Date', inplace=True)

# Save the cleaned DataFrame
df.to_csv("../data/AAPL_daily_cleaned.csv", index=False)

print("Cleaned data saved to 'data/AAPL_daily_cleaned.csv'")
