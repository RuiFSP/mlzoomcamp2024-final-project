# Use a slim python base image
FROM python:3.10-slim

# Install pipenv (optional, based on your workflow)
RUN pip install pipenv

# Set the working directory for the application
WORKDIR /app

# Copy Pipfile and Pipfile.lock
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install production dependencies from Pipfile
RUN pipenv install --system --deploy

# Copy application files into the container
COPY . /app/

# Ensure model files are also copied
COPY scripts/predict.py /app/scripts/predict.py
COPY data/processed/data_for_model.csv /app/data/processed/data_for_model.csv
COPY models /app/models/

# Set the environment variable for the Flask app
ENV FLASK_APP=app.py

# Expose the port the app will run on
EXPOSE 9696

# Use Gunicorn to run the Flask app
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "scripts.predict:app"]
