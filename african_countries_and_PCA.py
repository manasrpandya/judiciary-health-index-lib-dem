import pandas as pd
import geopandas as gpd
import folium
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# === Step 1: Load Data ===
# Load the V-Dem dataset
vdem_df = pd.read_csv("V-Dem.csv", low_memory=False)

# Filter for the year 2023
vdem_df = vdem_df[vdem_df["year"] == 2023]

# List of African countries
african_countries = [
    'Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi',
    'Cape Verde', 'Cameroon', 'Central African Republic', 'Chad', 'Comoros',
    'Democratic Republic of the Congo', 'Djibouti', 'Egypt',
    'Equatorial Guinea', 'Eritrea', 'Eswatini', 'Ethiopia',
    'Gabon', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Ivory Coast', 'Kenya', 'Lesotho',
    'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius',
    'Morocco', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Republic of the Congo', 'Rwanda',
    'Sao Tome and Principe', 'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 'Somaliland',
    'South Africa', 'South Sudan', 'Sudan', 'Tanzania', 'The Gambia', 'Togo', 'Tunisia',
    'Uganda', 'Zambia', 'Zimbabwe', 'Zanzibar'
]

# Select only African countries
vdem_df = vdem_df[vdem_df["country_name"].isin(african_countries)]

# List of judiciary indicators
indicators = [
    "v2juhcind", "v2juncind", "v2jucorrdc", "v2juaccnt", "v2cltrnslw",
    "v2juhccomp", "v2jucomp", "v2clacjstm", "v2clacjstw",
    "v2jureform", "v2jupurge", "v2jupoatck", "v2jupack", "v2clrspct"
]

# Drop rows with missing values
vdem_df = vdem_df.dropna(subset=indicators)

# === Step 2: Normalize Indicators by Their Max Ordinal Values ===
# Dictionary of ordinal scales
ordinal_scales = {
    "v2juhcind": 4, "v2juncind": 4, "v2jucorrdc": 4, "v2juaccnt": 4, 
    "v2cltrnslw": 4, "v2juhccomp": 4, "v2jucomp": 4, 
    "v2clacjstm": 4, "v2clacjstw": 4, "v2jureform": 2, "v2jupurge": 4, 
    "v2jupoatck": 4, "v2jupack": 3, "v2clrspct": 4
}

# Normalize each indicator by dividing by (max ordinal value + 1)
for ind in indicators:
    vdem_df[ind + "_normalized"] = vdem_df[ind] / (ordinal_scales[ind] + 1)

# === Step 3: Compute Weighted Index ===
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

# === Step 4: Perform PCA ===
scaler = StandardScaler()
X = scaler.fit_transform(vdem_df[[ind + "_normalized" for ind in indicators]])

pca = PCA(n_components=1)  # Extract only the first principal component
pca_index = pca.fit_transform(X)

# Store PCA results in DataFrame
vdem_df["pca_index"] = pca_index.flatten()

# Normalize PCA Index to [0,1]
vdem_df["pca_index"] = (vdem_df["pca_index"] - vdem_df["pca_index"].min()) / (
    vdem_df["pca_index"].max() - vdem_df["pca_index"].min()
)

# === Step 5: Create Choropleth Map for Africa ONLY ===
world = gpd.read_file("new/ne_110m_admin_0_countries.shp")
world = world[world["NAME_EN"].isin(african_countries)]
merged = world.merge(vdem_df[["country_name", "judiciary_health_index"]], left_on="NAME_EN", right_on="country_name", how="left")

m = folium.Map(location=[0, 20], zoom_start=4)

choropleth = folium.Choropleth(
    geo_data=merged,
    name="choropleth",
    data=merged,
    columns=["NAME_EN", "judiciary_health_index"],
    key_on="feature.properties.NAME_EN",
    fill_color="YlGnBu",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Judiciary Health Index (0-1)",
).add_to(m)

folium.GeoJsonTooltip(
    fields=["NAME_EN", "judiciary_health_index"],
    aliases=["Country:", "Judiciary Health Index:"],
    localize=True
).add_to(choropleth.geojson)

title_html = """
<div style="position: fixed; top: 20px; left: 20px; width: 300px; background-color: white; padding: 10px; 
    z-index: 9999; font-family: Arial, sans-serif; box-shadow: 2px 2px 10px rgba(0,0,0,0.3);">
    <h3 style="margin: 0; font-size: 18px; color: #333;">Judiciary Health Index (Africa)</h3>
    <p style="margin: 5px 0 0; font-size: 14px; color: #666;">
        Hover over a country to see its name and exact Judiciary Health Index score.
    </p>
    
    <p style="margin: 5px 0 0; font-size: 14px; color: #333;">
    NOTE: No country has indicator value above 0.29. Hence this choropleth shows colors only from 0 to 0.29
    </p>
</div>
"""
m.get_root().html.add_child(folium.Element(title_html))
m.save("africa_judiciary_healt.html")

# === Step 6: Plot PCA vs. Weighted Index ===
plt.figure(figsize=(8, 6))
sns.scatterplot(x=vdem_df["judiciary_health_index"], y=vdem_df["pca_index"], color="blue", alpha=0.7)

m, b = np.polyfit(vdem_df["judiciary_health_index"], vdem_df["pca_index"], 1)
plt.plot(vdem_df["judiciary_health_index"], m * vdem_df["judiciary_health_index"] + b, color="red")

plt.xlabel("Weighted Judiciary Index")
plt.ylabel("PCA-Based Judiciary Index")
plt.title("Comparison of PCA-Based Index vs Weighted Index")
plt.grid()
plt.savefig("pca_vs_weighted.png")  # Save PCA plot
plt.show()

pearson_corr, _ = pearsonr(vdem_df["judiciary_health_index"], vdem_df["pca_index"])
print(f"Pearson Correlation Coefficient: {pearson_corr:.4f}")

print("Map saved as 'africa_judiciary_healt.html' and PCA plot as 'pca_vs_weighted.png'.")
