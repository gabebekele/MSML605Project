FROM python:3.11

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

COPY requirements-m3.txt .

RUN pip install --upgrade pip
RUN pip install --verbose --index-url https://download.pytorch.org/whl/cpu torch torchvision
RUN pip install -r requirements-m3.txt

COPY . .

CMD ["python", "run_inference.py"]