---
title: GPT-2 Disease Website
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
startup_duration_timeout: 1h
short_description: Vietnamese GPT-2 disease text generation demo
models:
  - sanim05/GPT2-disease_text_generation
preload_from_hub:
  - sanim05/GPT2-disease_text_generation
pinned: false
---

# GPT-2 Disease Website

This project is a web interface for generating Vietnamese disease-related text with a fine-tuned GPT-2 model.

The frontend is built with React and Vite. The backend is a small Python HTTP server that loads the Hugging Face model `sanim05/GPT2-disease_text_generation` and exposes generation endpoints.

## Features

- Generate Vietnamese disease descriptions from a custom prompt
- Use built-in example topics such as cardiovascular, respiratory, digestive, and dermatology cases
- Adjust generation settings such as length, temperature, top-p, and repetition penalty
- Check backend availability from the UI before generating text

## Tech Stack

- Frontend: React 19, TypeScript, Vite
- Backend: Python, `transformers`, `torch`
- Model: `sanim05/GPT2-disease_text_generation`

## Project Structure

```text
.
├── src/                 # React frontend
├── backend/server.py    # Python inference server
├── backend/requirements.txt
├── package.json
└── vite.config.ts
```

## Run Locally

### 1. Install frontend dependencies

```bash
npm install
```

### 2. Set up the Python backend

```bash
npm run setup:backend
```

### 3. Start the backend

```bash
npm run dev:backend
```

The backend runs on `http://127.0.0.1:8000`.

### 4. Start the frontend

Open a second terminal and run:

```bash
npm run dev
```

The frontend runs on `http://127.0.0.1:3000`.

## Deploy Notes

Local development works because Vite proxies `/api` to `http://127.0.0.1:8000`.

If you deploy the frontend separately, `fetch('/api/generate')` will hit the frontend host itself. On platforms like Vercel, that returns a platform 404 unless you also deploy an API route on the same host.

For a separate backend deployment, set:

```bash
VITE_API_BASE_URL=https://your-backend-domain.com
```

Then the frontend will call:

- `https://your-backend-domain.com/api/health`
- `https://your-backend-domain.com/api/generate`

This repository's Python backend is a standalone HTTP server in `backend/server.py`. It is not automatically exposed as a Vercel `/api/*` function just by deploying the frontend.

## Backend Environment Variables

You can customize the backend with these optional variables:

- `BACKEND_HOST`: backend host, default `127.0.0.1`
- `BACKEND_PORT`: backend port, default `8000`
- `MODEL_ID`: Hugging Face model ID
- `MODEL_PATH`: local model path if you want to load weights from disk
- `MODEL_LOCAL_ONLY`: set to `1` to disable remote model downloads
- `PRELOAD_MODEL`: set to `0` to skip loading the model at startup

## Frontend Environment Variables

- `VITE_API_BASE_URL`: optional absolute backend origin for deployed frontend environments

## API Endpoints

- `GET /api/health`: health check and model status
- `POST /api/generate`: generate text from a prompt

Example request body:

```json
{
  "prompt": "viem phoi la",
  "max_new_tokens": 120,
  "temperature": 0.8,
  "top_p": 0.95,
  "repetition_penalty": 1.1
}
```

## Push Changes To GitHub

Your first push already succeeded. For the next updates, use:

```bash
git add .
git commit -m "Update README in English"
git push origin main
```

## Repository

GitHub: `https://github.com/Tuancoolboy/website-gpt2-disease.git`

## Deploy To Hugging Face Spaces

This repository is configured for a Hugging Face Docker Space.

### 1. Create a new Space

Create a new Space on Hugging Face and choose:

- SDK: `Docker`
- Visibility: your choice

### 2. Push this repository to the Space

Replace `YOUR_USERNAME` and `YOUR_SPACE_NAME`:

```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git push hf main
```

If you already have the `hf` remote:

```bash
git push hf main
```

### 3. Optional Space secrets / variables

Recommended:

- `HF_TOKEN`: your Hugging Face read token for more reliable Hub downloads

Optional:

- `MODEL_ID`
- `PRELOAD_MODEL=1`
- `MAX_NEW_TOKENS_LIMIT=128`

### 4. Result

When the Space finishes building:

- `/` serves the React frontend
- `/api/health` serves backend health
- `/api/generate` serves text generation
