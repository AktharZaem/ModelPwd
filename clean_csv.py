import pandas as pd

# Load the CSV file
input_file = '/Users/mohamedakthar/Desktop/Model Training/ModelAppPer/mobile_app_permission.csv'
df = pd.read_csv(input_file, header=0)  # Assuming first row is header

# Basic cleaning: strip whitespace, convert to lowercase for consistency
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Drop unnecessary columns if they exist
df = df.drop(columns=['Timestemp', 'ID', 'age'], errors='ignore')

# Standardize common variations in answers (example: normalize permission habits)
df['What is the best habit for managing app permissions on your phone?'] = df['What is the best habit for managing app permissions on your phone?'].replace({
    'give all apps full access': 'give all apps full access',
    'review and revoke unused permissions regularly': 'review and revoke unused permissions regularly',
    'restart your phone weekly': 'restart your phone weekly',
    'turn off notifications': 'turn off notifications'
    # Add more mappings as needed based on data inspection
})

# Handle any missing values (fill with 'unknown' or drop, here we fill)
df = df.fillna('unknown')

# Save the cleaned CSV
output_file = '/Users/mohamedakthar/Desktop/Model Training/ModelAppPer/mobile_app_permission_cleaned.csv'
df.to_csv(output_file, index=False)

print(f"Cleaned CSV saved to {output_file}")
