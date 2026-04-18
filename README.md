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

## Backend Environment Variables

You can customize the backend with these optional variables:

- `BACKEND_HOST`: backend host, default `127.0.0.1`
- `BACKEND_PORT`: backend port, default `8000`
- `MODEL_ID`: Hugging Face model ID
- `MODEL_PATH`: local model path if you want to load weights from disk
- `MODEL_LOCAL_ONLY`: set to `1` to disable remote model downloads
- `PRELOAD_MODEL`: set to `0` to skip loading the model at startup

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
