
import numpy as np
import cv2
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
import pandas as pd
import os

def analyze_vessels(binary_mask, output_prefix="results"):
    os.makedirs("results_vis", exist_ok=True)

    mask = (binary_mask > 0).astype(np.uint8)
    skeleton = skeletonize(mask).astype(np.uint8)
    labeled_mask = label(skeleton)
    props = regionprops(labeled_mask)
    distance_map = distance_transform_edt(mask)
    color_mask = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)

    vessel_data = []

    for i, prop in enumerate(props, start=1):
        coords = prop.coords
        length = len(coords)
        branch_type = 1 if length > 20 else 0

        thicknesses = [distance_map[y, x]*2 for y, x in coords]
        mean_thick = np.mean(thicknesses) if thicknesses else 0

        for y, x in coords:
            cv2.circle(color_mask, (x, y), 1, (0, 255, 0), -1)

        vessel_data.append({
            "Branch ID": i,
            "Length (px)": length,
            "Mean Thickness (px)": round(mean_thick, 2),
            "Type": branch_type
        })

    total_length = np.sum(skeleton)
    total_area = np.sum(mask)
    branching_points = np.sum(cv2.filter2D(skeleton, -1, np.ones((3, 3))) >= 4)
    thickness_mean = np.mean(distance_map[skeleton > 0]) * 2 if np.any(skeleton) else 0
    thickness_max = np.max(distance_map) * 2 if np.any(distance_map) else 0

    metrics = {
        "Vessel Area (px)": int(total_area),
        "Vessel Length (px)": int(total_length),
        "Branching Points": int(branching_points),
        "Mean Thickness (px)": round(thickness_mean, 2),
        "Max Thickness (px)": round(thickness_max, 2)
    }

    pd.DataFrame(vessel_data).to_csv(f"{output_prefix}_01_vessels.csv", index=False)
    pd.DataFrame([metrics]).to_csv(f"{output_prefix}.csv", index=False)
    out_path = os.path.join("results_vis", f"{output_prefix}.jpg")
    cv2.imwrite(out_path, color_mask)

    return metrics
