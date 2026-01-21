#
# Dockerfile for the PHASE Frontend (React + Tailwind)
# This is for PRODUCTION builds.
#

# --- Stage 1: The "Builder" ---
FROM node:18-alpine AS builder
WORKDIR /app

COPY package.json ./
COPY package-lock.json ./
RUN npm install

# Copy all source and config files
COPY . .
RUN npm run build

# --- Stage 2: The "Final Server" ---
FROM nginx:1.25-alpine
COPY --from=builder /app/build /usr/share/nginx/html

# Configure Nginx for a Single Page App (SPA)
RUN echo "server { \
      listen 3000; \
      location / { \
        root /usr/share/nginx/html; \
        index index.html index.htm; \
        try_files \$uri \$uri/ /index.html; \
      } \
    }" > /etc/nginx/conf.d/default.conf

EXPOSE 3000
