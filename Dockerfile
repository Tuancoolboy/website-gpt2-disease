FROM node:20-bookworm-slim AS frontend-build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY index.html metadata.json tsconfig.json vite.config.ts ./
COPY src ./src
RUN npm run build

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    PRELOAD_MODEL=1

RUN useradd -m -u 1000 user

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

COPY --chown=user requirements.txt ./requirements.txt
COPY --chown=user backend ./backend
COPY --chown=user server.py ./server.py
RUN pip install --user -r requirements.txt

COPY --from=frontend-build --chown=user /app/dist ./dist

EXPOSE 7860

CMD ["python", "server.py"]
