"""Phase 4: feature engineering - turn raw rows into fraud signals.

Golden rule: a feature may only use information available AT the moment
of the transaction (no peeking into the future = no data leakage).
"""
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

DB = Path("data/gold/fraudsentinel.db")
OUT = Path("data/gold/features.parquet")
BIG = 10**10                 # spacer so (customer, time) fits in one sortable number
HOUR, DAY = 3600, 86400


def to_seconds(s):
    return ((s - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).to_numpy()


def window_count(tx_c, tx_t, ev_c, ev_t, window):
    """For each transaction: how many events of the SAME customer happened
    in the last `window` seconds, up to and including now. Past only."""
    keys = np.sort(ev_c.astype(np.int64) * BIG + ev_t)
    now = tx_c.astype(np.int64) * BIG + tx_t
    return np.searchsorted(keys, now, side="right") - np.searchsorted(keys, now - window, side="left")


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def main():
    con = sqlite3.connect(DB)
    tx = pd.read_sql("SELECT txn_id, customer_id, ts, amount, category, merch_lat, merch_long, "
                     "is_fraud, split FROM transactions", con, parse_dates=["ts"])
    cust = pd.read_sql("SELECT customer_id, lat, long, state FROM customers", con)
    logins = pd.read_sql("SELECT l.customer_id, l.ts, l.state, l.success, d.first_seen "
                         "FROM logins l JOIN devices d ON l.device_id = d.device_id",
                         con, parse_dates=["ts", "first_seen"])
    changes = pd.read_sql("SELECT customer_id, ts FROM profile_changes", con, parse_dates=["ts"])

    tx = tx.merge(cust, on="customer_id").sort_values(["customer_id", "ts"]).reset_index(drop=True)
    code = lambda s: pd.Categorical(s, categories=cust["customer_id"]).codes
    tc, tt = code(tx.customer_id), to_seconds(tx.ts)
    f = pd.DataFrame({"txn_id": tx.txn_id})

    # ---------- A. TRANSACTION BEHAVIOUR ----------
    f["amount"] = tx.amount
    f["hour"] = tx.ts.dt.hour
    f["is_night"] = ((f.hour < 6) | (f.hour >= 22)).astype(int)
    f["is_online"] = tx.category.str.endswith("_net").astype(int)

    g = tx.groupby("customer_id")["amount"]
    prior_avg = (g.cumsum() - tx.amount) / g.cumcount()          # average of PAST purchases only
    f["amount_vs_avg"] = (tx.amount / prior_avg).replace([np.inf, -np.inf], np.nan).fillna(1.0).round(2)

    risk = tx[tx.split == "train"].groupby("category")["is_fraud"].mean()   # TRAIN only
    f["category_risk"] = tx.category.map(risk).fillna(risk.mean()).round(4)
    f["distance_km"] = haversine_km(tx.lat, tx.long, tx.merch_lat, tx.merch_long).round(1)

    # ---------- B. VELOCITY ----------
    f["txn_count_1h"] = window_count(tc, tt, tc, tt, HOUR) - 1   # minus the transaction itself
    f["txn_count_24h"] = window_count(tc, tt, tc, tt, DAY) - 1
    f["secs_since_prev_txn"] = tx.groupby("customer_id")["ts"].diff().dt.total_seconds().fillna(-1)

    # ---------- C. LOGIN & DEVICE ----------
    lc, lt = code(logins.customer_id), to_seconds(logins.ts)
    ok = logins.success.to_numpy() == 1
    is_new = ((logins.ts - logins.first_seen) <= pd.Timedelta(days=7)) & \
             (logins.first_seen > logins.first_seen.min())       # ignore day-1 home devices
    new_dev = ok & is_new.to_numpy()
    home_state = logins.customer_id.map(cust.set_index("customer_id")["state"])
    away = ok & (logins.state != home_state).to_numpy()
    fail = ~ok

    f["new_device_login_7d"] = window_count(tc, tt, lc[new_dev], lt[new_dev], 7 * DAY)
    f["away_login_7d"] = window_count(tc, tt, lc[away], lt[away], 7 * DAY)
    f["failed_logins_24h"] = window_count(tc, tt, lc[fail], lt[fail], DAY)
    cc, ct = code(changes.customer_id), to_seconds(changes.ts)
    f["address_change_7d"] = window_count(tc, tt, cc, ct, 7 * DAY)

    # ---------- SAVE ----------
    f["is_fraud"] = tx.is_fraud
    f["split"] = tx.split
    f.to_parquet(OUT, index=False)
    f.to_sql("features", con, if_exists="replace", index=False, chunksize=50_000)
    con.execute("CREATE INDEX IF NOT EXISTS idx_feat_txn ON features(txn_id)")
    gold = pd.read_sql("SELECT case_id, txn_id, scenario FROM golden_cases", con)
    con.commit()
    con.close()

    # ---------- SANITY CHECKS ----------
    cols = [c for c in f.columns if c not in ("txn_id", "is_fraud", "split")]
    summary = f[f.split == "train"].groupby("is_fraud")[cols].mean().T.round(3)
    summary.columns = ["legit_avg", "fraud_avg"]
    print(f"Built {len(cols)} features for {len(f):,} transactions -> {OUT}\n")
    print("=== Average feature value: legit vs fraud (train) ===")
    print(summary.to_string(), "\n")

    print("=== One golden case per scenario ===")
    view = gold.merge(f, on="txn_id")[["case_id", "scenario", "amount", "amount_vs_avg",
                                       "new_device_login_7d", "away_login_7d",
                                       "failed_logins_24h", "address_change_7d"]]
    print(view.groupby("scenario").head(1).to_string(index=False))


if __name__ == "__main__":
    main()