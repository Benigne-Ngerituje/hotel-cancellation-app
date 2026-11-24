# Hotel Booking Cancellation Predictor

A Flask web app that predicts the probability that a hotel booking will be cancelled.
A manager enters the booking details and gets:

- the **cancellation probability**
- a **risk level** (Low / Medium / High) with a suggested action
- the **top 3 reasons** behind the prediction (SHAP values)

Model: tuned LightGBM inside a scikit-learn pipeline (feature engineering, one-hot and target encoding, scaling), with a decision threshold chosen on the Precision-Recall curve.
Data: Kaggle – Hotel Booking Cancellation Prediction.

## Project structure

```
hotel-cancellation-app/
├── app.py              # Flask app: form, prediction, SHAP reasons, JSON API
├── features.py         # Feature engineering used inside the saved pipeline
├── model.joblib        # Trained pipeline + threshold (exported from the notebook)
├── requirements.txt    # Exact library versions, written by the notebook export cell
├── .python-version     # Python version, written by the notebook export cell
├── templates/          # HTML pages (base, form, result)
└── static/style.css    # Styling
```

## Risk levels

| Level  | Probability                               | Suggested action                         |
|--------|-------------------------------------------|------------------------------------------|
| Low    | below 30%                                 | Standard confirmation                    |
| Medium | 30% up to the decision threshold          | Reminder before arrival                  |
| High   | at or above the threshold (model: cancel) | Deposit / prepayment, plan overbooking   |

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000

## API

`POST /api/predict` with JSON, for example:

```json
{
  "reservation_date": "2026-07-15", "lead_time": 240,
  "no_of_weekend_nights": 2, "no_of_week_nights": 3,
  "no_of_adults": 2, "no_of_children": 0, "no_of_special_requests": 0,
  "parking_space": 0, "market_segment": "Online", "room_type": "Room_Type 1",
  "meal_plan": "Meal Plan 1", "avg_price": 140, "repeated_guest": 0,
  "prev_cancellations": 0, "prev_not_canceled": 0
}
```

`GET /health` returns the app status.

## Deployment (Render)

1. Push this folder to a public GitHub repository (including `model.joblib`).
2. On render.com: **New → Web Service**, connect the repository.
3. Settings:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app`
   - Instance type: Free
4. Click **Deploy**. The live link appears at the top of the service page.
