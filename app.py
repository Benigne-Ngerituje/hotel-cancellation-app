"""Hotel Booking Cancellation Predictor - Flask web app.

Loads the trained pipeline (model.joblib) and returns, for one booking:
the cancellation probability, a risk level, a suggested action and the
top 3 reasons behind the prediction (SHAP values from the model).
"""
import os
import calendar
from datetime import date

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

import features  # noqa: F401  (the saved pipeline needs features.prepare)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# ---------------------------------------------------------------- model
bundle = joblib.load(os.path.join(BASE_DIR, "model.joblib"))
PIPE = bundle["pipeline"]
THRESHOLD = float(bundle["threshold"])
FEATURE_STEP = PIPE.named_steps["features"]
PREP = PIPE.named_steps["prep"]
MODEL = PIPE.named_steps["model"]
TRANSFORMED_NAMES = list(PREP.get_feature_names_out())

BASE_COLS = [
    "no_of_adults", "no_of_children", "no_of_weekend_nights", "no_of_week_nights",
    "meal_plan", "parking_space", "room_type", "lead_time", "market_segment",
    "repeated_guest", "prev_cancellations", "prev_not_canceled", "avg_price",
    "no_of_special_requests", "reservation_month", "reservation_weekday",
]
# Use the exact column order the pipeline was trained with
_fitted = list(getattr(PIPE, "feature_names_in_", []))
INPUT_COLS = [c for c in _fitted if c in BASE_COLS] or BASE_COLS

MODEL_NAME = bundle.get("model_name", type(MODEL).__name__)
TEST_ROC_AUC = bundle.get("test_roc_auc")

# Dropdown options come from the categories the one-hot encoder learned
CAT_COLS, CATEGORIES, PREFIXES = [], {}, set()
for name, transformer, cols in PREP.transformers_:
    PREFIXES.add(name)
    if hasattr(transformer, "categories_") and type(transformer).__name__ == "OneHotEncoder":
        CAT_COLS = list(cols)
        CATEGORIES = {c: [str(v) for v in cats] for c, cats in zip(cols, transformer.categories_)}

OPTIONS = {
    "meal_plan": CATEGORIES.get("meal_plan", ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"]),
    "room_type": CATEGORIES.get("room_type", [f"Room_Type {i}" for i in range(1, 8)]),
    "market_segment": CATEGORIES.get("market_segment", ["Online", "Offline", "Corporate", "Aviation", "Complementary"]),
}

# Risk levels: High = the model predicts a cancellation (probability >= threshold)
LOW_CUT = min(0.30, THRESHOLD / 2)
RISK_LEVELS = {
    "Low": "Standard confirmation is enough.",
    "Medium": "Send a reminder a few days before arrival and confirm the guest's plans.",
    "High": "Ask for a deposit or prepayment, and include this booking in overbooking plans.",
}

# Model features grouped into business-friendly reasons
GROUPS = {
    "lead_time": "Lead time", "lead_time_group": "Lead time",
    "no_of_special_requests": "Special requests", "has_special_request": "Special requests",
    "avg_price": "Room price", "price_per_guest": "Room price",
    "market_segment": "Market segment",
    "no_of_adults": "Number of guests", "no_of_children": "Number of guests",
    "total_guests": "Number of guests", "has_children": "Number of guests",
    "no_of_weekend_nights": "Length of stay", "no_of_week_nights": "Length of stay",
    "total_nights": "Length of stay", "weekend_share": "Length of stay",
    "total_stay_cost": "Room price", "is_free_booking": "Room price",
    "prior_bookings": "Previous cancellations", "prev_cancel_rate": "Previous cancellations",
    "reservation_month": "Reservation month", "reservation_weekday": "Reservation weekday",
    "meal_plan": "Meal plan", "room_type": "Room type", "parking_space": "Car parking",
    "repeated_guest": "Repeat guest",
    "prev_cancellations": "Previous cancellations", "has_prev_cancellation": "Previous cancellations",
    "prev_not_canceled": "Previous completed stays",
}

FIELDS = [
    # name, label, min, max
    ("no_of_adults", "Adults", 0, 10),
    ("no_of_children", "Children", 0, 10),
    ("no_of_weekend_nights", "Weekend nights", 0, 30),
    ("no_of_week_nights", "Week nights", 0, 60),
    ("lead_time", "Lead time (days)", 0, 1000),
    ("no_of_special_requests", "Special requests", 0, 10),
    ("prev_cancellations", "Previous cancellations", 0, 100),
    ("prev_not_canceled", "Previous completed stays", 0, 100),
]

DEFAULTS = {
    "reservation_date": date.today().isoformat(),
    "no_of_adults": 2, "no_of_children": 0,
    "no_of_weekend_nights": 1, "no_of_week_nights": 2,
    "meal_plan": OPTIONS["meal_plan"][0], "room_type": OPTIONS["room_type"][0],
    "market_segment": "Online" if "Online" in OPTIONS["market_segment"] else OPTIONS["market_segment"][0],
    "lead_time": 30, "avg_price": 100, "no_of_special_requests": 0,
    "parking_space": 0, "repeated_guest": 0, "prev_cancellations": 0, "prev_not_canceled": 0,
}


# ---------------------------------------------------------------- input
def parse_booking(data):
    """Validate raw form/JSON input. Returns (row, errors)."""
    errors, row = [], {}

    for name, label, lo, hi in FIELDS:
        try:
            value = int(float(data.get(name, "")))
            if not lo <= value <= hi:
                raise ValueError
            row[name] = value
        except (TypeError, ValueError):
            errors.append(f"{label} must be a whole number between {lo} and {hi}.")

    try:
        price = float(data.get("avg_price", ""))
        if not 0 <= price <= 10000:
            raise ValueError
        row["avg_price"] = price
    except (TypeError, ValueError):
        errors.append("Average price per night must be a number between 0 and 10,000.")

    for name in ("parking_space", "repeated_guest"):
        row[name] = 1 if str(data.get(name, "0")).lower() in ("1", "yes", "true", "on") else 0

    for name, options in OPTIONS.items():
        value = str(data.get(name, ""))
        if value not in options:
            errors.append(f"Please choose a valid {name.replace('_', ' ')}.")
        row[name] = value

    try:
        d = date.fromisoformat(str(data.get("reservation_date", "")))
        row["reservation_month"] = d.month
        row["reservation_weekday"] = d.weekday()      # 0 = Monday, same as training
    except ValueError:
        errors.append("Please enter a valid reservation date.")

    # Same rules as the data cleaning step
    if not errors:
        if row["no_of_adults"] + row["no_of_children"] < 1:
            errors.append("A booking needs at least one guest.")
        if row["no_of_weekend_nights"] + row["no_of_week_nights"] < 1:
            errors.append("A booking needs at least one night.")

    return row, errors


# ---------------------------------------------------------------- prediction
def original_feature(transformed_name):
    """'num__lead_time' -> 'lead_time', 'cat__market_segment_Online' -> 'market_segment'."""
    prefix, _, rest = transformed_name.partition("__")
    for col in sorted(CAT_COLS, key=len, reverse=True):
        if rest.startswith(col + "_"):
            return col
    return rest


def contributions(Xp):
    """SHAP contributions per transformed feature (LightGBM or XGBoost)."""
    Xp = np.asarray(Xp, dtype=float)
    if hasattr(MODEL, "booster_"):                       # LightGBM
        values = MODEL.predict(Xp, pred_contrib=True)
    else:                                                # XGBoost
        import xgboost as xgb
        values = MODEL.get_booster().predict(xgb.DMatrix(Xp), pred_contribs=True)
    return np.asarray(values)[0][:-1]                    # last column = base value


def describe(group, row):
    """Human-readable value of a reason group for this booking."""
    if group == "Lead time":
        return f"{row['lead_time']} days"
    if group == "Special requests":
        return str(row["no_of_special_requests"])
    if group == "Room price":
        return f"{row['avg_price']:.2f} per night"
    if group == "Market segment":
        return row["market_segment"]
    if group == "Number of guests":
        return f"{row['no_of_adults']} adults, {row['no_of_children']} children"
    if group == "Length of stay":
        return f"{row['no_of_weekend_nights']} weekend + {row['no_of_week_nights']} week nights"
    if group == "Reservation month":
        return calendar.month_name[row["reservation_month"]]
    if group == "Reservation weekday":
        return calendar.day_name[row["reservation_weekday"]]
    if group == "Meal plan":
        return row["meal_plan"]
    if group == "Room type":
        return row["room_type"].replace("_", " ")
    if group == "Car parking":
        return "Yes" if row["parking_space"] else "No"
    if group == "Repeat guest":
        return "Yes" if row["repeated_guest"] else "No"
    if group == "Previous cancellations":
        return str(row["prev_cancellations"])
    if group == "Previous completed stays":
        return str(row["prev_not_canceled"])
    return ""


def predict(row):
    X = pd.DataFrame([row])[INPUT_COLS]
    prob = float(PIPE.predict_proba(X)[0, 1])

    if prob >= THRESHOLD:
        risk = "High"
    elif prob >= LOW_CUT:
        risk = "Medium"
    else:
        risk = "Low"

    # SHAP contributions straight from the model
    contribs = contributions(PREP.transform(FEATURE_STEP.transform(X)))

    totals = {}
    for name, value in zip(TRANSFORMED_NAMES, contribs):
        group = GROUPS.get(original_feature(name), original_feature(name))
        totals[group] = totals.get(group, 0.0) + float(value)

    top = sorted(totals.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
    reasons = [{
        "factor": group,
        "value": describe(group, row),
        "effect": "increases" if impact > 0 else "lowers",
        "impact": round(impact, 3),
    } for group, impact in top]

    return {
        "probability": round(prob, 4),
        "risk_level": risk,
        "action": RISK_LEVELS[risk],
        "threshold": round(THRESHOLD, 2),
        "reasons": reasons,
    }


# ---------------------------------------------------------------- routes
@app.context_processor
def model_info():
    return {"model_name": MODEL_NAME, "test_roc_auc": TEST_ROC_AUC}


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        row, errors = parse_booking(request.form)
        if not errors:
            result = predict(row)
            return render_template("result.html", result=result, row=row, low_cut=LOW_CUT,
                                   month=calendar.month_name[row["reservation_month"]],
                                   weekday=calendar.day_name[row["reservation_weekday"]])
        return render_template("index.html", values=request.form, errors=errors, options=OPTIONS)
    return render_template("index.html", values=DEFAULTS, errors=[], options=OPTIONS)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    row, errors = parse_booking(data)
    if errors:
        return jsonify({"errors": errors}), 400
    return jsonify(predict(row))


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME, "threshold": THRESHOLD})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
