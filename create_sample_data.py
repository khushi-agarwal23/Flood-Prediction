import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

os.makedirs("data", exist_ok=True)

grid_size = 50
days = 100

rows = []
start_date = datetime(2020,1,1)

for day in range(days):

    current_date = start_date + timedelta(days=day)

    for r in range(grid_size):
        for c in range(grid_size):

            rainfall = np.random.gamma(2, 10)  # realistic rainfall distribution

            risk = np.random.choice(
                ["SAFE","MODERATE","HIGH","CRITICAL"],
                p=[0.7,0.2,0.08,0.02]
            )

            rows.append({
                "date": current_date,
                "day": day,
                "zone_id": r*grid_size + c,
                "grid_row": r,
                "grid_col": c,

                "rainfall_mm": rainfall,

                "risk": risk,
                "final_degradation": np.random.uniform(0,0.30),

                "land_use": np.random.choice([
                    "residential_light",
                    "residential_dense",
                    "commercial",
                    "industrial",
                    "mixed_use",
                    "green_space"
                ]),

                "flood_event": np.random.choice([0,1], p=[0.95,0.05]),
                "load_ratio": np.random.uniform(0.2,1.2),
                "drift_memory": np.random.uniform(0,0.1),
                "degradation_factor": np.random.uniform(0,0.30),

                "w1": np.random.uniform(0.2,0.4),
                "w2": np.random.uniform(0.2,0.4),
                "w3": np.random.uniform(0.2,0.4)
            })

df = pd.DataFrame(rows)

df.to_csv("data/simulation_10yr.csv", index=False)
df.to_csv("data/simulation_5yr.csv", index=False)

print("Sample data created:", len(df), "rows")