FROM python:3.9-slim-buster

RUN pip install pandas pytz openpyxl

COPY config /config
COPY scripts /scripts

WORKDIR /scripts

ENTRYPOINT ["/bin/sh", "-c" , "python imaging_occurrence.py && python imaging_feature.py && python measurement.py"]
