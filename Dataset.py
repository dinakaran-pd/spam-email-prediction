

import pandas as pd
import os


def load_dataset(path="spam.csv"):
    """
    Loads the dataset from `path` and returns a DataFrame with
    columns: ['text', 'label']  where label is 0 (ham) or 1 (spam).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'. Put your dataset CSV in the "
            f"project folder and update the filename if needed."
        )

    # Some spam CSVs have messy rows (extra unescaped commas inside the
    # message text), which breaks pandas' fast default parser. We try
    # the normal fast read first, and if that fails, fall back to a
    # more forgiving parser that skips broken rows instead of crashing.
    try:
        df = pd.read_csv(path, encoding="latin-1")
    except pd.errors.ParserError:
        print("Warning: some rows in the CSV are malformed "
              "(inconsistent number of columns). Re-reading and "
              "skipping those bad rows...")
        df = pd.read_csv(
            path,
            encoding="latin-1",
            engine="python",
            on_bad_lines="skip",
        )

    # Drop fully-empty "Unnamed" columns some CSV exports include
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Normalize column names to lowercase for easy matching
    original_cols = list(df.columns)
    df.columns = [c.strip().lower() for c in df.columns]

    # --- Detect known column naming schemes ---
    if "v1" in df.columns and "v2" in df.columns:
        # Classic "SMS Spam Collection" dataset format
        df = df.rename(columns={"v1": "label", "v2": "text"})

    elif "category" in df.columns and "message" in df.columns:
        df = df.rename(columns={"category": "label", "message": "text"})

    elif "class" in df.columns and "text" in df.columns:
        df = df.rename(columns={"class": "label"})

    elif "label" in df.columns and "text" in df.columns:
        pass  # already in the right shape

    elif "label" in df.columns and "message" in df.columns:
        df = df.rename(columns={"message": "text"})

    elif "label" in df.columns and "email" in df.columns:
        df = df.rename(columns={"email": "text"})

    elif "text" in df.columns and "spam" in df.columns:
        df = df.rename(columns={"spam": "label"})

    else:
        raise ValueError(
            f"Couldn't recognize the columns {original_cols}. "
            f"Please tell me your actual column names so I can update "
            f"dataset.py to map them correctly."
        )

    # Keep only what we need, drop empty rows
    df = df[["text", "label"]].dropna()

    # Convert label to 0/1 no matter how it's currently written
    # (handles "spam"/"ham", "spam"/"not spam", 1/0, "1"/"0", etc.)
    def to_binary(val):
        val = str(val).strip().lower()
        if val in ("spam", "1", "1.0", "true"):
            return 1
        return 0

    df["label"] = df["label"].apply(to_binary)

    # Reset index after all the filtering above
    df = df.reset_index(drop=True)

    return df


if __name__ == "__main__":
    # Quick manual test: run "python dataset.py" to sanity-check loading
    data = load_dataset("spam.csv")
    print(data.head())
    print("\nTotal rows:", len(data))
    print("Spam count:", data["label"].sum())
    print("Ham count:", (data["label"] == 0).sum())