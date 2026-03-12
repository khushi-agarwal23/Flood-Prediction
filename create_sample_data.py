import pandas as pd
import numpy as np

grid_size = 50
days = 100

zones = []

for r in range(grid_size):
    for c in range(grid_size):
        zones.append({
            "zone_id": r * grid_size + c,
            "grid_row": r,
            "grid_col": c,
            "land_use": np.random.choice(["urban","industrial","residential","green"])
        })

zones_df = pd.DataFrame(zones)

rows = []

for day in range(days):
    for _, z in zones_df.iterrows():

        rainfall = np.random.gamma(2, 10)
        degradation = min(0.3, 0.002 * day + np.random.uniform(0,0.02))
        if rainfall > 60:
            risk = "CRITICAL"
        elif rainfall > 40:
            risk = "HIGH"
        elif rainfall > 20:
            risk = "MODERATE"
        else:
            risk = "SAFE"

        rows.append({
            "zone_id": z.zone_id,
            "grid_row": z.grid_row,
            "grid_col": z.grid_col,
            "land_use": z.land_use,
            "day": day,
            "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=day),
            "rainfall_mm": rainfall,
            "degradation_factor": degradation,
            "risk": risk
        })

sim = pd.DataFrame(rows)

sim_10yr = sim
sim_5yr = sim[sim["day"] < days/2]

zones_df.to_csv("data/city_zones.csv", index=False)
sim_10yr.to_csv("data/simulation_10yr.csv", index=False)
sim_5yr.to_csv("data/simulation_5yr.csv", index=False)

print("10yr rows:", len(sim_10yr))
print("5yr rows:", len(sim_5yr))
