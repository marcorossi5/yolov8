###############
# builder stage
###############
FROM python:3.10-slim as builder

SHELL ["/bin/bash", "-c"]

RUN apt update && apt upgrade

ENV POETRY_VERSION=1.5.1
ENV VENV=/opt/venv
ENV POETRY_VIRTUALENVS_CREATE=false

RUN python3 -m venv $VENV
ENV PATH="${VENV}/bin:${PATH}"

# install poetry
RUN apt update && apt upgrade -y && apt install -y \
    ffmpeg \
    libsm6 \
    libxext6

RUN pip install --upgrade pip && pip install "poetry==$POETRY_VERSION"

# copy dependencies
ARG PYPROJECT_FILE
COPY $PYPROJECT_FILE /tmp

# install dependencies
RUN poetry run -C /tmp pip install --upgrade pip
RUN poetry install --no-interaction --no-root -C /tmp

###############
# prod stage
###############

FROM python:3.10-slim as prod

WORKDIR /home

# copy lib files
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# copy source files
COPY ui/ ui/

COPY yolo/ yolo/

EXPOSE 8000
CMD ["streamlit", "run", "ui/app.py", "--server.port", "8000", "--server.address", "0.0.0.0"]
