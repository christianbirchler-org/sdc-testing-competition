FROM python:3.11

# Expose PORT
EXPOSE $PORT

# Create a non-root user
RUN addgroup --gid 1000 app && \
    adduser --uid 1000 --ingroup app --home /home/app --shell /bin/sh --disabled-password --gecos "" app

# Add user installed pip packages to path
ENV PATH /home/app/.local/bin:$PATH

# Switch to less privileged user
USER ${uid}:${gid}

# Set working directory
WORKDIR /home/app/selector

RUN pip install --user --upgrade pip --no-cache-dir

# Setup dependencies and dependency caching
COPY --chown=app:app requirements.txt ./requirements.txt
RUN pip install --user --requirement requirements.txt --no-cache-dir

# Copy source code
COPY --chown=app:app . .

# Configure Python environment
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

ENTRYPOINT ["python", "main.py"]
