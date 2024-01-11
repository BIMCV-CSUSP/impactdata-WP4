import os
import pandas as pd
import SimpleITK as sitk
import pydicom
import datetime
import cv2
from pathlib import Path
from radiomics import featureextractor


if __name__ == "__main__":
    extractor = featureextractor.RadiomicsFeatureExtractor(geometryTolerance=1.0)
    input_dir = Path("/input")
    masks_dir = input_dir.joinpath("derivatives", "lung_segmentation")
    radiomics_dir = input_dir.joinpath("derivatives", "radiomics")
    images = input_dir.glob("*/*/*/*/*.png")

    for filename in images:
        print(f"Procesando la imagen {filename}")
        mod = filename.stem.split("_")[-1]
        mask_filename = masks_dir.joinpath(
            filename.relative_to(input_dir).parent.joinpath(
                filename.stem.replace(mod, f"mod-{mod}_seg.png")
            )
        )
        print(mask_filename)

        try:
            image = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
            sitk_image = sitk.GetImageFromArray(image)
        except Exception as e:
            print(f"Error al leer el pixel array: {e}")
            continue
        try:
            mask = cv2.imread(str(mask_filename), cv2.IMREAD_GRAYSCALE)
            sitk_mask = sitk.GetImageFromArray(mask)
        except Exception as e:
            print(f"Error al leer el pixel array: {e}")
            continue

        try:
            result = extractor.execute(sitk_image, sitk_mask)
        except Exception as e:
            print("Error {} extrayendo features de imagen {}".format(e, str(filename)))
            continue

        # Save the results that starts with "original_" in a dataframe
        df = pd.DataFrame.from_dict(
            {k: v for k, v in result.items() if k.startswith("original_")},
            orient="index",
        )
        # Transpose the dataframe to have the features as columns
        df = df.transpose()
        # Delete the original_ prefix from the column names
        df.columns = df.columns.str.replace("original_", "")
        # Save the dataframe as a csv file
        radiomics_filename = radiomics_dir.joinpath(
            filename.relative_to(input_dir).parent.joinpath(
                filename.stem + "_radiomics.csv"
            )
        )
        radiomics_filename.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(radiomics_filename, index=False)
        print("Guardando radiomics en {}".format(radiomics_filename))
