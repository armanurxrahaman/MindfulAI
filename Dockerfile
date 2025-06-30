FROM python:3.11

WORKDIR /code

COPY backend/api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/api/main.py ./main.py
COPY backend/models ./models
COPY backend/improved_facial_model.py ./improved_facial_model.py
COPY backend/models ./models
COPY backend/services ./services
COPY backend/routes ./routes

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"] 