# Urdu Bot

![Urdu Bot](assets/UI.png)

## Overview

Urdu Bot is an Urdu-language chatbot that utilizes the Llama3 model through Ollama, featuring speech-to-text conversation capabilities. This bot serves as an anxiety therapist, providing support and assistance with real-time anxiety level monitoring.

## Features

- Text Input
- Speech Input
- Real Time Anxiety Meter
- Text To Speech

## Directory Structure

```
├── README.md
├── app.py
├── audio_files
│   ├── 06-14-2024-19-22-10.wav
│   ├── 06-14-2024-19-22-19.wav
│   └── ...
├── config.py
├── database.py
├── instance
│   └── <your database instance>
├── model
│   └── <xlm-roberta-urdu>
├── requirements.txt
├── rough.ipynb
├── static
│   ├── css
│   │   └── styles.css
│   └── js
│       └── app.js
├── templates
│   └── chat.html
└── utils
    ├── __init__.py
    ├── preprocessing.py
    └── stop_words.py
```

## Usage

### Setup Environment

Tested on Python 3.10.

```bash
conda create -n <env_name> python==3.10
conda activate <env_name>
```

### Install Ollama

Install Ollama by running the following command:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Run Llama3 with Ollama

Start Ollama and download the Llama3 model:

```bash
ollama serve & ollama pull llama3
```

Ollama will run in the background at `http://127.0.0.1:11434/`.

### Setup Urdu Bot

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Run

Start the flask application:

```bash
flask run
```

or

```bash
python app.py
```

You're now ready to use **Urdu Bot**! 🎉

---
