"""Phase 2: Bronze -> Silver -> Gold pipeline for FraudSentinel."""
import hashlib, hmac, os, sqlite3
from pathlib import Path
import pandas as pd

BRONZE, SILVER, GOLD = Path("data/bronze"), Path("data/silver"), Path("data/gold")
KEY = os.environ.get("FS_TOKEN_KEY", "dev-only-change-me").encode()


# ---------- helper ----------
def tokenize(prefix: str, value) -> str:
    """Same input + same key -> same token. Can't be reversed without the key."""
    digest = hmac.new(KEY, str(value).encode(), hashlib.sha256).hexdigest()
    return f"{prefix}_{digest[:12]}"


# ---------- BRONZE -> SILVER ----------
def bronze_to_silver() -> pd.DataFrame:
    # 1. Load both raw files and combine them
    frames = []
    for file, split in [("fraudTrain.csv", "train"), ("fraudTest.csv", "test")]:
        df = pd.read_csv(BRONZE / file, index_col=0)   # index_col=0 drops the unnamed column
        df["split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # 2. Fix data types (text -> real dates)
    df["ts"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])

    # 3. PCI: replace card numbers with tokens
    cards = {c: tokenize("card", c) for c in df["cc_num"].unique()}
    df["card_id"] = df["cc_num"].map(cards)

    # 4. Create a stable customer ID
    person = df["first"] + "|" + df["last"] + "|" + df["dob"].astype(str)
    custs = {p: tokenize("cust", p) for p in person.unique()}
    df["customer_id"] = person.map(custs)

    # 5. Remove the misleading "fraud_" prefix from merchant names
    df["merchant"] = df["merchant"].str.removeprefix("fraud_")
    mers = {m: tokenize("mer", m) for m in df["merchant"].unique()}
    df["merchant_id"] = df["merchant"].map(mers)

    # 6. Drop the raw card number and redundant columns
    df = df.drop(columns=["cc_num", "trans_date_trans_time", "unix_time"])

    # 7. Save as Parquet
    SILVER.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SILVER / "transactions.parquet", index=False)
    return df


# ---------- SILVER -> GOLD ----------
def silver_to_gold(df: pd.DataFrame) -> None:
    # 1. Split one big table into 4 clean tables
    customers = df[["customer_id", "first", "last", "gender", "street", "city", "state",
                     "zip", "lat", "long", "city_pop", "job", "dob"]].drop_duplicates("customer_id")
    customers = customers.assign(dob=customers["dob"].dt.strftime("%Y-%m-%d"))

    cards = df[["card_id", "customer_id"]].drop_duplicates("card_id")

    merchants = df[["merchant_id", "merchant", "category"]].drop_duplicates("merchant_id")

    txns = df[["trans_num", "card_id", "customer_id", "merchant_id", "ts", "amt",
               "category", "merch_lat", "merch_long", "is_fraud", "split"]]
    txns = txns.rename(columns={"trans_num": "txn_id", "amt": "amount"})
    txns = txns.assign(ts=txns["ts"].dt.strftime("%Y-%m-%d %H:%M:%S"))

    # 2. Write them to SQLite
    GOLD.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(GOLD / "fraudsentinel.db") as con:
        for name, table in [("customers", customers), ("cards", cards),
                            ("merchants", merchants), ("transactions", txns)]:
            table.to_sql(name, con, if_exists="replace", index=False, chunksize=50_000)

        # 3. Indexes for fast lookups
        con.execute("CREATE INDEX IF NOT EXISTS idx_txn_cust_ts ON transactions(customer_id, ts)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_txn_card_ts ON transactions(card_id, ts)")


# ---------- RUN ----------
if __name__ == "__main__":
    df = bronze_to_silver()
    silver_to_gold(df)
    print(f"Transactions: {len(df):,}")
    print(f"Customers:    {df['customer_id'].nunique():,}")
    print(f"Cards:        {df['card_id'].nunique():,}")
    print(f"Merchants:    {df['merchant_id'].nunique():,}")
    print(f"Fraud rate:   {df['is_fraud'].mean():.2%}")