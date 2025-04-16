import time
import numpy as np
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, MimeType, BBox, CRS

# Configure Sentinel Hub credentials
config = SHConfig()
config.sh_client_id = "e4642559-5bd2-4767-acc2-9ab3102cb631"  # Replace with your client ID
config.sh_client_secret = "AYBP7gYzLb2eTNdNdPsD7j5JMPmXdSXG"  # Replace with your client secret

if not config.sh_client_id or not config.sh_client_secret:
    raise ValueError("Sentinel Hub credentials are not set. Please set SH client ID and secret.")

def get_satellite_data(lat, lon):
    """
    Fetch average NDVI value for given latitude and longitude using Sentinel Hub Process API.
    Retries up to 3 times on failure.
    """
    retries = 3
    for attempt in range(retries):
        try:
            # Define a small bounding box (~1 km) around the point
            point_bbox = BBox(bbox=[lon - 0.005, lat - 0.005, lon + 0.005, lat + 0.005], crs=CRS.WGS84)

            # Evalscript with SCL masking (clouds, water, shadows)
            evalscript = """
            //VERSION=3
            function setup() {
                return {
                    input: ["B04", "B08", "SCL"],
                    output: { bands: 1, sampleType: "FLOAT32" }
                };
            }
            function evaluatePixel(sample) {
                // Mask out water (6), cloud shadows (3), clouds (8,9)
                if (sample.SCL === 3 || sample.SCL === 6 || sample.SCL === 8 || sample.SCL === 9) {
                    return [NaN];
                }
                let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
                return [ndvi];
            }
            """

            request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A,
                        time_interval=("2023-01-01", "2023-12-31"),
                        mosaicking_order='leastCC'
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response('default', MimeType.TIFF)
                ],
                bbox=point_bbox,
                size=(64, 64),
                config=config
            )

            data = request.get_data()

            if not data:
                raise ValueError("No data returned from Sentinel Hub.")

            ndvi_array = data[0]  # 2D array of NDVI values
            valid_ndvi = ndvi_array[np.isfinite(ndvi_array)]

            if valid_ndvi.size == 0:
                raise ValueError("No valid NDVI values found.")

            return float(np.mean(valid_ndvi))

        except Exception as e:
            print(f"Attempt {attempt + 1} failed for lat={lat}, lon={lon}: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise e
