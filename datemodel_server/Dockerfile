FROM python:3.7-slim

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY * /app/

CMD cd /app && python3 main.py
