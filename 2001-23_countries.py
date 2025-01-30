import pandas as pd
import matplotlib.pyplot as plt

# === Step 1: Load Data ===
# Load the V-Dem dataset
vdem_df = pd.read_csv("V-Dem.csv", low_memory=False)

# Filter for the required years (2001-2023)
vdem_df = vdem_df[(vdem_df["year"] >= 2001) & (vdem_df["year"] <= 2023)]

# List of selected countries
selected_countries = ["India", "Tanzania", "Zanzibar", "Japan", "Germany"]

# Filter for selected countries
vdem_df = vdem_df[vdem_df["country_name"].isin(selected_countries)]

# List of judiciary-related indicators
indicators = [
    "v2juhcind", "v2juncind", "v2jucorrdc", "v2juaccnt", "v2cltrnslw",
    "v2juhccomp", "v2jucomp", "v2clacjstm", "v2clacjstw",
    "v2jureform", "v2jupurge", "v2jupoatck", "v2jupack", "v2clrspct"
]

# === Step 2: Check for Missing Values ===
# Print any missing values
missing_values = vdem_df[indicators].isnull().sum()
print("Missing values per indicator:")
print(missing_values[missing_values > 0])

# Drop rows where any indicator is missing
vdem_df = vdem_df.dropna(subset=indicators)

# === Step 3: Normalize Indicators Properly ===
# Load mean values
mean_values = vdem_df[indicators].mean()

# Default normalization factors
normalization_factors = {
    "v2juhcind": 5, "v2juncind": 5, "v2jucorrdc": 5, "v2juaccnt": 5, 
    "v2cltrnslw": 5, "v2juhccomp": 5, "v2jucomp": 5, 
    "v2clacjstm": 5, "v2clacjstw": 5, "v2jureform": 3, "v2jupurge": 5, 
    "v2jupoatck": 5, "v2jupack": 4, "v2clrspct": 5
}

# Normalize each indicator
for ind in indicators:
    vdem_df[ind + "_normalized"] = vdem_df[ind] / normalization_factors[ind]
    vdem_df[ind + "_normalized"] = vdem_df[ind + "_normalized"].clip(0, 1)  # Ensure values are between 0 and 1

# === Step 4: Compute Weighted Index ===
# Define weights
weights = {
    "v2juhcind": 0.12, "v2juncind": 0.12, "v2jucorrdc": 0.08, "v2juaccnt": 0.08,
    "v2cltrnslw": 0.08, "v2juhccomp": 0.08, "v2jucomp": 0.07, 
    "v2clacjstm": 0.06, "v2clacjstw": 0.06, "v2jureform": 0.06, "v2jupurge": 0.06,
    "v2jupoatck": 0.05, "v2jupack": 0.05, "v2clrspct": 0.05
}

# Compute the weighted sum for the new index
vdem_df["judiciary_health_index"] = sum(
    vdem_df[ind + "_normalized"] * weight for ind, weight in weights.items()
)

# Ensure the final index is between 0 and 1
vdem_df["judiciary_health_index"] = vdem_df["judiciary_health_index"].clip(0, 1)

# === Step 5: Create a Line Chart ===
plt.figure(figsize=(12, 6))

# Plot for each country
for country in selected_countries:
    country_df = vdem_df[vdem_df["country_name"] == country]
    plt.plot(country_df["year"], country_df["judiciary_health_index"], marker="o", label=country)

# Formatting the chart
plt.xlabel("Year")
plt.ylabel("Judiciary Health Index (0-1)")
plt.title("Judiciary Health Index (2001-2023)")
plt.legend()
plt.grid(True)

# Show the chart
plt.savefig("2001-23.png")  