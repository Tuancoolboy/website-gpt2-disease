# GPT-2 Disease Website

This project is a web application for generating Vietnamese disease-related text with the fine-tuned GPT-2 model `sanim05/GPT2-disease_text_generation`. It lets users start from a short prompt such as a disease name or opening sentence, then asks the model to continue that text into a longer description.

The goal of the project is to provide a simple interface for experimenting with controlled medical text generation in Vietnamese. Users can choose a sample topic, rewrite the opening prompt, and adjust generation settings such as output length, tone, idea coverage, and repetition control before sending the request to the backend.

The frontend is built with React and Vite, while the backend is a lightweight Python server that loads the model and exposes generation endpoints. Together, they provide a complete demo for testing prompt-based disease description generation in a browser.

## Run Locally

```bash
npm install
npm run setup:backend
npm run dev:backend
```

Open another terminal and run:

```bash
npm run dev
```

- Frontend: `http://127.0.0.1:3000`
- Backend: `http://127.0.0.1:8000`
