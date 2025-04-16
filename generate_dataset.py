import csv
import random
import os
from sentinel_api import get_satellite_data

# Define dataset path
dataset_path = "data/dataset.csv"

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

# Parameters
desired_samples = 100  # Number of valid samples you want
max_attempts = 500     # Max total attempts to avoid infinite loops

# Latitude range to exclude poles (approximate land range)
min_lat, max_lat = -60, 60

valid_samples = 0
attempts = 0

# Open log file for missing NDVI data
missing_log_path = "data/missing_ndvi.log"
missing_log = open(missing_log_path, "a")

with open(dataset_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["latitude", "longitude", "ndvi", "temperature", "crowd_density"])  # Headers

    while valid_samples < desired_samples and attempts < max_attempts:
        lat = random.uniform(min_lat, max_lat)
        lon = random.uniform(-180, 180)
        attempts += 1

        try:
            ndvi = get_satellite_data(lat, lon)  # Fetch NDVI from Sentinel Hub

            # Validate NDVI range; assign default if invalid
            if not (-1.0 <= ndvi <= 1.0):
                print(f"Invalid NDVI {ndvi} at lat={lat}, lon={lon}, skipping sample")
                missing_log.write(f"Invalid NDVI {ndvi} for lat={lat}, lon={lon}\n")
                continue

            temp = random.uniform(15, 40)  # Simulated temperature in °C
            crowd_density = random.randint(0, 100)  # Simulated crowd density (raw scale)

            writer.writerow([lat, lon, ndvi, temp, crowd_density])
            print(f"Saved: lat={lat:.4f}, lon={lon:.4f}, ndvi={ndvi:.4f}, temp={temp:.2f}, crowd={crowd_density}")

            valid_samples += 1

        except Exception as e:
            print(f"Error fetching NDVI data for lat={lat:.4f}, lon={lon:.4f}: {e}")
            missing_log.write(f"Missing NDVI data for lat={lat}, lon={lon}: {e}\n")

missing_log.close()
print(f"Dataset saved at {dataset_path}")
print(f"Missing NDVI data logged at {missing_log_path}")
print(f"Total attempts: {attempts}, Valid samples: {valid_samples}")
