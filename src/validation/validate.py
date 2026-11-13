"""
Data Validation Module.
Validates schemas, data types, value boundaries, null constraints,
duplicate records, and referential integrity between tables.
Invalid records are isolated into data/errors/, and a validation report is generated.
"""

import json
from datetime import datetime
from pathlib import Path
import pandas as pd


def validate_datasets(
    raw_dir: str = "data/raw",
    errors_dir: str = "data/errors",
    reports_dir: str = "reports"
) -> dict:
    """
    Performs comprehensive data quality checks on ratings, items, and users.
    Saves invalid records to errors_dir and writes an automated report to reports_dir.
    """
    raw_path = Path(raw_dir)
    errors_path = Path(errors_dir)
    reports_path = Path(reports_dir)

    errors_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)

    ratings_file = raw_path / "ratings.csv"
    items_file = raw_path / "items.csv"
    users_file = raw_path / "users.csv"

    if not (ratings_file.exists() and items_file.exists() and users_file.exists()):
        raise FileNotFoundError("Raw files missing. Run ingestion first.")

    df_ratings = pd.read_csv(ratings_file)
    df_items = pd.read_csv(items_file)
    df_users = pd.read_csv(users_file)

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append(f"DATA VALIDATION REPORT - {datetime.utcnow().isoformat()}Z")
    report_lines.append("=" * 70)

    # -------------------------------------------------------------
    # 1. Validate Users Dataset
    # -------------------------------------------------------------
    report_lines.append("\n[1] VALIDATING USERS DATASET")
    initial_user_count = len(df_users)
    report_lines.append(f"Total raw user records: {initial_user_count}")

    # Null checks
    user_nulls = df_users["user_id"].isna()
    # Age range check (1 to 120)
    user_invalid_age = (~df_users["age"].isna()) & ((df_users["age"] < 1) | (df_users["age"] > 120))
    # Duplicate user_id check
    user_duplicates = df_users.duplicated(subset=["user_id"], keep="first")

    invalid_user_mask = user_nulls | user_invalid_age | user_duplicates
    invalid_users = df_users[invalid_user_mask].copy()

    if not invalid_users.empty:
        invalid_users["error_reason"] = "user_id_null" * user_nulls[invalid_user_mask] + \
                                       ";invalid_age" * user_invalid_age[invalid_user_mask] + \
                                       ";duplicate_user_id" * user_duplicates[invalid_user_mask]
        invalid_users_file = errors_path / "invalid_users.csv"
        invalid_users.to_csv(invalid_users_file, index=False)
        report_lines.append(f"  ❌ Found {len(invalid_users)} invalid user records -> Saved to {invalid_users_file}")
    else:
        report_lines.append("  ✅ User schema, unique constraints, and age bounds: PASSED (0 errors)")

    valid_users = df_users[~invalid_user_mask]
    valid_user_ids = set(valid_users["user_id"])

    # -------------------------------------------------------------
    # 2. Validate Items Dataset
    # -------------------------------------------------------------
    report_lines.append("\n[2] VALIDATING ITEMS DATASET")
    initial_item_count = len(df_items)
    report_lines.append(f"Total raw item records: {initial_item_count}")

    item_nulls = df_items["item_id"].isna() | df_items["title"].isna()
    item_duplicates = df_items.duplicated(subset=["item_id"], keep="first")

    invalid_item_mask = item_nulls | item_duplicates
    invalid_items = df_items[invalid_item_mask].copy()

    if not invalid_items.empty:
        invalid_items["error_reason"] = "item_id_or_title_null" * item_nulls[invalid_item_mask] + \
                                       ";duplicate_item_id" * item_duplicates[invalid_item_mask]
        invalid_items_file = errors_path / "invalid_items.csv"
        invalid_items.to_csv(invalid_items_file, index=False)
        report_lines.append(f"  ❌ Found {len(invalid_items)} invalid item records -> Saved to {invalid_items_file}")
    else:
        report_lines.append("  ✅ Item schema and uniqueness: PASSED (0 errors)")

    valid_items = df_items[~invalid_item_mask]
    valid_item_ids = set(valid_items["item_id"])

    # -------------------------------------------------------------
    # 3. Validate Ratings & Referential Integrity
    # -------------------------------------------------------------
    report_lines.append("\n[3] VALIDATING RATINGS & REFERENTIAL INTEGRITY")
    initial_ratings_count = len(df_ratings)
    report_lines.append(f"Total raw interaction records: {initial_ratings_count}")

    # Boundary checks
    rating_nulls = df_ratings[["user_id", "item_id", "rating", "timestamp"]].isna().any(axis=1)
    rating_bounds = (df_ratings["rating"] < 1.0) | (df_ratings["rating"] > 5.0)
    timestamp_bounds = (df_ratings["timestamp"] < 700000000) | (df_ratings["timestamp"] > 2500000000)
    rating_duplicates = df_ratings.duplicated(subset=["user_id", "item_id"], keep=False)

    # Referential Integrity: User ID and Item ID must exist in parent tables
    user_ref_integrity = ~df_ratings["user_id"].isin(valid_user_ids)
    item_ref_integrity = ~df_ratings["item_id"].isin(valid_item_ids)

    invalid_ratings_mask = (
        rating_nulls |
        rating_bounds |
        timestamp_bounds |
        user_ref_integrity |
        item_ref_integrity
    )

    invalid_ratings = df_ratings[invalid_ratings_mask].copy()
    if not invalid_ratings.empty:
        reasons = []
        for idx in invalid_ratings.index:
            r = []
            if rating_nulls.loc[idx]: r.append("missing_values")
            if rating_bounds.loc[idx]: r.append("rating_out_of_bounds_1_5")
            if timestamp_bounds.loc[idx]: r.append("invalid_timestamp")
            if user_ref_integrity.loc[idx]: r.append("foreign_key_user_not_found")
            if item_ref_integrity.loc[idx]: r.append("foreign_key_item_not_found")
            reasons.append(";".join(r))
        invalid_ratings["error_reason"] = reasons
        invalid_ratings_file = errors_path / "invalid_ratings.csv"
        invalid_ratings.to_csv(invalid_ratings_file, index=False)
        report_lines.append(f"  ❌ Found {len(invalid_ratings)} invalid interaction records -> Saved to {invalid_ratings_file}")
    else:
        report_lines.append("  ✅ Rating ranges [1.0, 5.0], timestamp bounds, and foreign key constraints: PASSED (0 errors)")

    report_lines.append(f"  ℹ️ Found {rating_duplicates.sum()} duplicate user-item pairs (will be handled during preparation).")

    valid_ratings_count = initial_ratings_count - len(invalid_ratings)
    report_lines.append(f"\nFinal Valid Interaction Records: {valid_ratings_count:,} ({valid_ratings_count/initial_ratings_count*100:.2f}%)")
    report_lines.append("=" * 70)

    report_content = "\n".join(report_lines)
    report_file = reports_path / "data_validation_report.txt"
    with open(report_file, "w") as f:
        f.write(report_content)

    print(report_content)

    return {
        "status": "success",
        "valid_ratings_count": valid_ratings_count,
        "invalid_ratings_count": len(invalid_ratings),
        "report_file": str(report_file)
    }


if __name__ == "__main__":
    validate_datasets()

