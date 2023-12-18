import os
import csv
import pandas as pd
import json
from pathlib import Path

input_dir = Path("/input")
omop_tables_dir = input_dir.joinpath("derivatives", "omop_tables")
config_file = Path("/config", "IDs.json")

with open(config_file) as f:
    config = json.load(f)

imaging_occurrence_id_start = config["imaging_occurrence_id_start"]
imaging_feature_id_start = config["imaging_feature_id_start"]
imaging_feature_domain_id_start = config["measurement_id_start"]

imaging_feature_id = -1
imaging_feature_domain_id = -1

patient_dirs = [folder for folder in omop_tables_dir.iterdir() if folder.is_dir()] # subject

for patient_dir in patient_dirs:
    patient_dir_path = omop_tables_dir.joinpath(patient_dir)

    procedure_dirs = [folder for folder in patient_dir_path.iterdir() if folder.is_dir()] # session

    for index, procedure_dir in enumerate(procedure_dirs, start=1):
        procedure_dir_path = patient_dir_path.joinpath(procedure_dir, "mod-rx")
        
        if imaging_feature_id != -1:
            imaging_feature_id_start = imaging_feature_id + 1
        if imaging_feature_domain_id != -1:
            imaging_feature_domain_id_start = imaging_feature_domain_id + 1

        rows = []
        print(procedure_dir_path)
        img_occ_filename = list(procedure_dir_path.glob("*imaging_occurrence.csv"))[0]
        with open(img_occ_filename, newline='') as img_occ_file:
            ids_reader = csv.DictReader(img_occ_file, delimiter=',')
 
            for row in ids_reader:
                imaging_occurrence_id = row['imaging_occurrence_id']

        for i, j in zip(range(imaging_feature_id_start, imaging_feature_id_start + 111), range(imaging_feature_domain_id_start, imaging_feature_domain_id_start + 111)):
            imaging_feature_id = i
            imaging_feature_domain_id = j

            row = {
                'imaging_feature_id': imaging_feature_id,
                'imaging_finding_num': 0,  #dx timepoint
                'imaging_occurrence_id': int(imaging_occurrence_id),
                'domain_concept_id': 21,  #measurement table code
                'imaging_feature_domain_id': imaging_feature_domain_id,
                'anatomic_site_location': '2000000026',  #change this value according to the vocabulary
                #TODO: define with IMPaCT group:
                'alg_system': 'https://github.com',  #example 
                'alg_datetime': '16-06-2023'  #example
            }

            rows.append(row)

        csv_file = img_occ_filename.parent.joinpath(img_occ_filename.stem.replace("imaging_occurrence", "imaging_feature.csv"))
        fieldnames = rows[0].keys()

        with open(csv_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"The file '{csv_file}' has been successfully created.")

        df = pd.DataFrame(rows)

        excel_file = img_occ_filename.parent.joinpath(img_occ_filename.stem.replace("imaging_occurrence", "imaging_feature.xlsx"))
        df.to_excel(excel_file, index=False)

        print(f"The file '{excel_file}' has been successfully created.")
