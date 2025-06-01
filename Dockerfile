# Use official Python image
FROM python:3.10-slim


# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

 
# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory in the container
WORKDIR /app

# Copy all files to the container
COPY . .


# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port your app runs on (adjust if different)
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]
