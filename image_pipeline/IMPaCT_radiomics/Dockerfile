FROM python:3.9

WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY extraer_features.py .

ENTRYPOINT [ "python", "/app/extraer_features.py" ]

