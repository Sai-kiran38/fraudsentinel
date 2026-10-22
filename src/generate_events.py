"""Phase 2b: synthetic devices, logins and profile changes for FraudSentinel.

Design rule: signals must be NOISY. Legit customers also get new phones,
travel and move; some fraud has no login signal. Otherwise
'new device = fraud' becomes a trivial rule.
"""
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

DB = Path("data/gold/fraudsentinel.db")
rng = np.random.default_rng(42)          # fixed seed -> identical data every run
DEVICE_TYPES = [("mobile", "iOS"), ("mobile", "Android"), ("desktop", "Windows"), ("desktop", "macOS")]


def make_device(cust_id, first_seen, n):
    dtype, os_ = DEVICE_TYPES[rng.integers(len(DEVICE_TYPES))]
    return {"device_id": f"dev_{cust_id[5:]}_{n:02d}", "customer_id": cust_id,
            "first_seen": first_seen, "device_type": dtype, "os": os_}


def login(cust_id, device_id, ts, city, state, success=1):
    return {"customer_id": cust_id, "device_id": device_id, "ts": ts,
            "city": city, "state": state, "success": success}


def routine_logins(cust_id, device_id, start, end, city, state, every_days=5):
    """Normal behaviour: a login every ~5 days, mostly daytime, from home."""
    span = max((end - start).days, 1)
    n = max(1, span // every_days)
    days = np.floor(rng.uniform(0, span, n))
    hours = rng.normal(14, 4, n).clip(6, 23)
    stamps = start.normalize() + pd.to_timedelta(days, "D") + pd.to_timedelta(hours, "h")
    return [login(cust_id, device_id, t, city, state) for t in stamps if start <= t <= end]


def random_time(start, end):
    return start + (end - start) * rng.random()


def main():
    con = sqlite3.connect(DB)
    customers = pd.read_sql("SELECT customer_id, city, state FROM customers", con)
    txns = pd.read_sql("SELECT customer_id, ts, is_fraud FROM transactions", con, parse_dates=["ts"])
    start, end = txns["ts"].min(), txns["ts"].max()
    first_fraud = txns[txns.is_fraud == 1].groupby("customer_id")["ts"].min().to_dict()
    places = customers[["city", "state"]].drop_duplicates().to_numpy()

    devices, logins, changes, scenarios = [], [], [], []

    for c in customers.itertuples():
        cid = c.customer_id

        # 1. NORMAL LIFE: 1-2 home devices with routine logins
        for i in range(rng.integers(1, 3)):
            d = make_device(cid, start, i)
            devices.append(d)
            logins += routine_logins(cid, d["device_id"], start, end, c.city, c.state)
        home_dev = f"dev_{cid[5:]}_00"

        # 2. LEGIT NOISE: this is what makes the problem hard
        if rng.random() < 0.30:                                  # new phone
            t = random_time(start, end)
            d = make_device(cid, t, 9)
            devices.append(d)
            logins += routine_logins(cid, d["device_id"], t, end, c.city, c.state)
            scenarios.append((cid, "legit_new_device", t))

        if rng.random() < 0.15:                                  # travel
            t = random_time(start, end)
            city, state = places[rng.integers(len(places))]
            for k in range(rng.integers(2, 6)):
                logins.append(login(cid, home_dev, t + pd.Timedelta(hours=12 * k), city, state))
            scenarios.append((cid, "legit_travel", t))

        if rng.random() < 0.05:                                  # legit address change
            t = random_time(start, end)
            city, state = places[rng.integers(len(places))]
            changes.append({"customer_id": cid, "ts": t, "field": "address",
                            "old_value": f"{c.city}, {c.state}", "new_value": f"{city}, {state}"})
            scenarios.append((cid, "legit_address_change", t))

        if rng.random() < 0.10:                                  # typo
            logins.append(login(cid, home_dev, random_time(start, end), c.city, c.state, success=0))

        # 3. FRAUD STORIES
        if cid in first_fraud:
            f = first_fraud[cid]
            if rng.random() < 0.45:                              # account takeover
                t_login = f - pd.Timedelta(minutes=int(rng.integers(2, 16)))
                d = make_device(cid, t_login, 99)
                devices.append(d)
                city, state = places[rng.integers(len(places))]
                for k in range(rng.integers(0, 4)):              # failed attempts first
                    logins.append(login(cid, d["device_id"], t_login - pd.Timedelta(minutes=k + 1),
                                        city, state, success=0))
                logins.append(login(cid, d["device_id"], t_login, city, state))
                if rng.random() < 0.6:
                    changes.append({"customer_id": cid, "ts": t_login + pd.Timedelta(minutes=1),
                                    "field": "address", "old_value": f"{c.city}, {c.state}",
                                    "new_value": f"{city}, {state}"})
                scenarios.append((cid, "fraud_account_takeover", t_login))
            else:                                                # stolen card details, no login
                scenarios.append((cid, "fraud_stolen_card_no_login", f))

    # ---------- write to gold ----------
    fmt = "%Y-%m-%d %H:%M:%S"
    dev = pd.DataFrame(devices)
    dev["first_seen"] = pd.to_datetime(dev["first_seen"]).dt.strftime(fmt)

    log = pd.DataFrame(logins).sort_values("ts").reset_index(drop=True)
    log["ts"] = pd.to_datetime(log["ts"]).dt.strftime(fmt)
    log.insert(0, "login_id", [f"lg_{i:07d}" for i in range(1, len(log) + 1)])

    chg = pd.DataFrame(changes).sort_values("ts").reset_index(drop=True)
    chg["ts"] = pd.to_datetime(chg["ts"]).dt.strftime(fmt)
    chg.insert(0, "change_id", [f"pc_{i:05d}" for i in range(1, len(chg) + 1)])

    # Ground truth for EVALUATION ONLY. The agent must never read this table.
    sc = pd.DataFrame(scenarios, columns=["customer_id", "scenario", "ts"])
    sc["ts"] = pd.to_datetime(sc["ts"]).dt.strftime(fmt)

    for name, table in [("devices", dev), ("logins", log),
                        ("profile_changes", chg), ("eval_scenarios", sc)]:
        table.to_sql(name, con, if_exists="replace", index=False, chunksize=50_000)
    con.execute("CREATE INDEX IF NOT EXISTS idx_login_cust_ts ON logins(customer_id, ts)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_chg_cust_ts ON profile_changes(customer_id, ts)")
    con.commit()
    con.close()

    # ---------- sanity check: is 'new device' a noisy signal? ----------
    counts = sc["scenario"].value_counts()
    legit_new = counts.get("legit_new_device", 0)
    fraud_new = counts.get("fraud_account_takeover", 0)
    print(f"Devices:          {len(dev):,}")
    print(f"Logins:           {len(log):,}  (failed: {(log.success == 0).sum():,})")
    print(f"Profile changes:  {len(chg):,}")
    print(counts.to_string())
    print(f"\n'New device' rule precision: {fraud_new / (legit_new + fraud_new):.0%}"
          f"  (good if well below 100%: the signal is noisy)")


if __name__ == "__main__":
    main()