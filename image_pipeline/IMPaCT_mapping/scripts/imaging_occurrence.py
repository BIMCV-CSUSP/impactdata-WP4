import os
import csv
import datetime
import pandas as pd
import json
from pathlib import Path

input_dir = Path("/input")
dicom_headers_dir = input_dir.joinpath("derivatives", "dicom_headers")
omop_tables_dir = input_dir.joinpath("derivatives", "omop_tables")
config_file = Path("/config", "IDs.json")

with open(config_file) as f:
    config = json.load(f)

imaging_occurrence_id_start = config["imaging_occurrence_id_start"]
imaging_occurrence_id = -1

patient_dirs = [folder for folder in input_dir.iterdir() if (folder.is_dir() and "sub" in str(folder))] # subject

print("Patient dirs: ")
print(patient_dirs)

for patient_dir in patient_dirs:
    patient_dir_path = input_dir.joinpath(patient_dir)
    print(f"Patient dir {patient_dir}")

    procedure_dirs = [folder for folder in patient_dir_path.iterdir() if folder.is_dir()] # session
    print(f"Procedure dirs {procedure_dirs}")
    print(imaging_occurrence_id)

    if imaging_occurrence_id != -1:
        imaging_occurrence_id_start = imaging_occurrence_id + 1
    
    for index, procedure_dir in enumerate(procedure_dirs, start=1):
        imaging_occurrence_id = imaging_occurrence_id_start + index - 1
        procedure_dir_path = patient_dir_path.joinpath(procedure_dir)

        print ("datos:" + str(imaging_occurrence_id) + ":" + str(procedure_dir) + ":" + str(procedure_dir_path))

        wadors_uri = procedure_dir_path.joinpath("mim-ligth")
        dicom_headers_files = wadors_uri.glob("*/*.json")
        rows = []

        index_image=1
        for dicom_headers_file in dicom_headers_files:
            imaging_occurrence_id_mids=(imaging_occurrence_id)*100+index_image
            index_image=index_image+1
            with open(dicom_headers_file, "r") as dicom_file:
                dicom_reader = json.load(dicom_file)
                imaging_occurrence_date = dicom_reader.get("00080021", {}).get("Value", datetime.date.today().strftime("%Y%m%d"))
                if isinstance(imaging_occurrence_date, list):
                    imaging_occurrence_date = imaging_occurrence_date[0]
                imaging_study_UID = dicom_reader.get("00080018", {}).get("Value", 0)
                if isinstance(imaging_study_UID, list):
                    imaging_study_UID = imaging_study_UID[0]
                imaging_series_UID = imaging_study_UID #ad-hoc for rx images

                dicom_tags = {}
                dicom_tags["imaging_occurrence_date"] = imaging_occurrence_date
                dicom_tags["imaging_study_uid"] = imaging_study_UID
                view_position = dicom_reader.get("00185101", {}).get("Value", 0)
                if isinstance(view_position, list):
                    view_position = view_position[0]
                dicom_tags["view_position"] = view_position
                spatial_resolution = dicom_reader.get("00181050", {}).get("Value", 0)
                if isinstance(spatial_resolution, list):
                    spatial_resolution = spatial_resolution[0]
                dicom_tags["spatial_resolution"] = spatial_resolution
                columns = dicom_reader.get("00280011", {}).get("Value", 0)
                if isinstance(columns, list):
                    columns = columns[0]
                dicom_tags["columns"] = columns
                dcm_rows = dicom_reader.get("00280010", {}).get("Value", 0)
                if isinstance(dcm_rows, list):
                    dcm_rows = dcm_rows[0]
                dicom_tags["rows"] = dcm_rows

            df = pd.DataFrame.from_dict(dicom_tags, orient="index")
            df = df.transpose()
            dicom_headers_path = dicom_headers_dir.joinpath(
            dicom_headers_file.relative_to(input_dir).parent.joinpath(
                    dicom_headers_file.stem + "_dicom_tags.csv"
                )
            )
            dicom_headers_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(dicom_headers_path, index=False)

            row = {
                'imaging_occurrence_id': imaging_occurrence_id_mids,
                'person_id': patient_dir.name,
                'procedure_occurrence_id': procedure_dir.name,
                'wadors_uri': str(wadors_uri),
                'imaging_occurrence_date': str(imaging_occurrence_date),
                'imaging_study_UID': str(imaging_study_UID),
                'imaging_series_UID': str(imaging_series_UID),
                'modality': 'RX',
                'anatomic_site_location': '2000000026'  #change this value according to the vocabulary
            }

            rows.append(row)

        fieldnames = rows[0].keys()

        omop_table_filename = omop_tables_dir.joinpath(
            dicom_headers_file.relative_to(input_dir).parent.joinpath(
                dicom_headers_file.stem + "_imaging_occurrence.csv"
            )
        )
        omop_table_filename.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(rows)

        omop_table_filename_excel = omop_tables_dir.joinpath(
            dicom_headers_file.relative_to(input_dir).parent.joinpath(
                dicom_headers_file.stem + "_imaging_occurrence.xlsx"
            )
        )
        df.to_excel(omop_table_filename_excel, index=False)

        print(f"The file '{omop_table_filename_excel}' has been successfully created.")

        df.to_csv(omop_table_filename, index=False)
        print(f"The file '{omop_table_filename}' has been successfully created.")
