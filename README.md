# AI/ML Model Selection Agent

A Flask + Scikit-learn web application that analyzes a CSV dataset, detects classification or regression, preprocesses features, evaluates multiple models, and recommends the best model among those tested.

## Run locally

```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

Install:
```bash
pip install -r requirements.txt
```

Run:
```bash
python app.py
```

Open:
http://127.0.0.1:5000

## Render deployment

Push this project to GitHub.

In Render:
1. Create a new Web Service.
2. Connect the GitHub repository.
3. Build Command:
   `pip install -r requirements.txt`
4. Start Command:
   `gunicorn app:app`
5. Deploy.

The application reads Render's PORT environment variable and listens on `0.0.0.0`.

## Important

Only CSV files are accepted. No database, hardware, or paid AI API is required.
