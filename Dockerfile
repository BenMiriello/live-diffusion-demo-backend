FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir pydantic-settings==2.0.3

# Copy the application
COPY . .

# Expose the API port
EXPOSE 8000

# Run the application
CMD ["python", "run.py"]
