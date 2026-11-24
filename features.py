
import numpy as np
import pandas as pd

LEAD_TIME_GROUPS = ["0-7 days", "8-30 days", "31-90 days", "91+ days"]


def safe_divide(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.zeros_like(a), where=(b != 0))


def prepare(data, price_cap=231.0):
    X = data.copy()

    # Outlier cap (same value as in cleaning)
    X["avg_price"] = X["avg_price"].clip(upper=price_cap)

    # Size and length of the stay
    X["total_guests"] = X["no_of_adults"] + X["no_of_children"]
    X["total_nights"] = X["no_of_weekend_nights"] + X["no_of_week_nights"]
    X["weekend_share"] = safe_divide(X["no_of_weekend_nights"], X["total_nights"])
    X["has_children"] = (X["no_of_children"] > 0).astype(int)

    # Price features
    X["price_per_guest"] = safe_divide(X["avg_price"], X["total_guests"])
    X["total_stay_cost"] = X["avg_price"] * X["total_nights"]
    X["is_free_booking"] = (X["avg_price"] == 0).astype(int)

    # Engagement and guest history
    X["has_special_request"] = (X["no_of_special_requests"] > 0).astype(int)
    X["prior_bookings"] = X["prev_cancellations"] + X["prev_not_canceled"]
    X["prev_cancel_rate"] = safe_divide(X["prev_cancellations"], X["prior_bookings"])
    X["has_prev_cancellation"] = (X["prev_cancellations"] > 0).astype(int)

    # How far ahead the booking was made
    X["lead_time_group"] = pd.cut(
        X["lead_time"], bins=[-1, 7, 30, 90, np.inf], labels=LEAD_TIME_GROUPS
    ).astype(str)
    return X
