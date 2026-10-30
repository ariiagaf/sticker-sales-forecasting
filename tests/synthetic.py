"""Synthetic data with the same schema as Kaggle train.csv (for tests only).

NEVER use it for reported results: numbers on this data mean nothing.
"""
import itertools

import numpy as np
import pandas as pd

COUNTRIES = ["Canada", "Finland", "Italy", "Kenya", "Norway", "Singapore"]
STORES = ["Discount Stickers", "Premium Sticker Mart", "Stickers for Less"]
PRODUCTS = [
    "Holographic Goose", "Kaggle", "Kaggle Tiers", "Kerneler", "Kerneler Dark Edition",
]


def make_synthetic_train(start="2010-01-01", end="2016-12-31", seed=0, missing=True):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end)
    t = np.arange(len(dates))
    frames = []
    for country, store, product in itertools.product(COUNTRIES, STORES, PRODUCTS):
        level = rng.uniform(80, 900)
        season = 1 + 0.3 * np.sin(2 * np.pi * dates.day_of_year / 365.25 + rng.uniform(0, 6))
        weekly = 1 + 0.1 * (dates.day_of_week >= 5)
        trend = 1 + 0.02 * t / 365
        y = level * season * weekly * trend * rng.lognormal(0, 0.05, len(dates))
        if country == "Norway" and product == "Kaggle Tiers":
            y = y * np.where(dates.year >= 2015, 0.7, 1.0)
        frames.append(pd.DataFrame({
            "date": dates, "country": country, "store": store,
            "product": product, "num_sold": np.maximum(1, np.round(y)),
        }))
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    df.insert(0, "id", np.arange(len(df)))
    if missing:
        mask = df["country"].isin(["Canada", "Kenya"]) & (df["product"] == "Holographic Goose")
        drop = mask & (rng.random(len(df)) < 0.5)
        df.loc[drop, "num_sold"] = np.nan
    return df
