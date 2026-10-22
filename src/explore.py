"""Phase 3: explore the data + build the golden test set."""
import sqlite3
from pathlib import Path
import pandas as pd

DB = Path("data/gold/fraudsentinel.db")
COLS = ["txn_id", "customer_id", "ts", "amount", "category", "is_fraud"]


# ---------- PART A: EXPLORE ----------
def explore(tx):
    print("=== 1. Class balance by split ===")
    print(tx.groupby("split")["is_fraud"].agg(txns="count", frauds="sum", rate="mean").to_string(), "\n")

    print("=== 2. Amount: legit (0) vs fraud (1) ===")
    print(tx.groupby("is_fraud")["amount"].describe()[["50%", "mean", "max"]].round(2).to_string(), "\n")

    print("=== 3. Highest fraud-rate categories ===")
    rate = tx.groupby("category")["is_fraud"].mean().sort_values(ascending=False).head(5)
    print(rate.map("{:.2%}".format).to_string(), "\n")

    print("=== 4. Fraud rate by time of day ===")
    band = pd.cut(tx["ts"].dt.hour, [-1, 5, 11, 17, 23],
                  labels=["night 00-05", "morning 06-11", "afternoon 12-17", "evening 18-23"])
    print(tx.groupby(band, observed=True)["is_fraud"].mean().map("{:.2%}".format).to_string(), "\n")

    print("=== 5. The accuracy trap ===")
    print(f"A model that ALWAYS says 'not fraud' is {1 - tx['is_fraud'].mean():.2%} accurate "
          f"and catches 0 fraud.\n")


# ---------- PART B: GOLDEN TEST SET ----------
def pick(df, n, expected, scenario, why):
    s = df.sample(min(n, len(df)), random_state=7)[COLS]
    return s.assign(expected=expected, scenario=scenario, why=why)


def build_golden(tx, sc):
    parts = []

    # 1. FRAUD: the first fraud transaction of each fraud story
    for scen, n, why in [
        ("fraud_account_takeover", 8, "New device + failed logins + maybe address change just before purchase"),
        ("fraud_stolen_card_no_login", 8, "No login signal: must be caught from transaction behaviour"),
    ]:
        custs = sc.loc[sc.scenario == scen, "customer_id"]
        first = (tx[(tx.is_fraud == 1) & tx.customer_id.isin(custs)]
                 .sort_values("ts").groupby("customer_id").head(1))
        parts.append(pick(first, n, "FRAUD", scen, why))

    # 2. TRAPS: legit purchases within 48h after a suspicious-looking legit event
    legit = tx[tx.is_fraud == 0]
    for scen, n, why in [
        ("legit_new_device", 6, "Customer got a new phone: new device but innocent"),
        ("legit_travel", 4, "Customer travelling: login from another city but innocent"),
    ]:
        ev = sc.loc[sc.scenario == scen, ["customer_id", "ts"]].rename(columns={"ts": "event_ts"})
        m = legit.merge(ev, on="customer_id")
        m = m[(m.ts > m.event_ts) & (m.ts <= m.event_ts + pd.Timedelta(hours=48))]
        first = m.sort_values("ts").groupby("customer_id").head(1)
        parts.append(pick(first, n, "LEGIT", scen, why))

    # 3. TRAP: big but genuine purchases
    parts.append(pick(legit[legit.amount > 1000], 4, "LEGIT", "legit_high_amount",
                      "Large purchase that is genuinely the customer's"))

    return pd.concat(parts, ignore_index=True)


# ---------- RUN ----------
def main():
    con = sqlite3.connect(DB)
    tx = pd.read_sql("SELECT txn_id, customer_id, ts, amount, category, is_fraud, split FROM transactions",
                     con, parse_dates=["ts"])
    sc = pd.read_sql("SELECT * FROM eval_scenarios", con, parse_dates=["ts"])

    explore(tx)

    golden = build_golden(tx, sc)
    golden.insert(0, "case_id", [f"GC-{i:03d}" for i in range(1, len(golden) + 1)])
    golden["ts"] = golden["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    golden.to_sql("golden_cases", con, if_exists="replace", index=False)
    con.close()

    print("=== 6. Golden test set ===")
    print(golden.groupby(["expected", "scenario"]).size().to_string())
    print(f"\nTotal: {len(golden)} cases saved to table golden_cases\n")
    print(golden[["case_id", "amount", "category", "expected", "scenario"]].head(3).to_string(index=False))


if __name__ == "__main__":
    main()