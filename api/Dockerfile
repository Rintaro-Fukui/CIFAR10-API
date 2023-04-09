FROM python:3.9-alpine

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN apk add --no-cache build-base \
    && pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt \
    && apk del build-base
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY ./src/ .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]