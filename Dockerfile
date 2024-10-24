FROM python:3.11

# Update and install necessary dependencies
RUN apt update && \
    apt install -y \
    postgresql \
    gcc 

RUN pip install --upgrade pip
RUN pip install poetry
RUN poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock ./
RUN poetry install

COPY . .