FROM node:20-slim AS base

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN python3 -m pip install --break-system-packages -r requirements.txt

# Install Node dependencies
COPY package.json package-lock.json ./
RUN npm ci

# Copy source code
COPY . .

# Build Next.js
RUN npm run build

EXPOSE 3000

ENV PYTHON_PATH=python3
ENV PORT=3000

CMD ["npm", "start"]
