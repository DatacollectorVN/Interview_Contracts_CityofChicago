"""
Author: Nathan Ngo
Created at: 23-07-2025

Full explanations for those steps are desribed in notebook `part4.ipynb`.
"""
import pandas as pd
import duckdb
import argparse
from typing import Optional


def _handle_null_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy() # Prevent adverse effects
    df["Contract Type"] = df["Contract Type"].fillna("Unknown")
    df["Department"] = df["Department"].fillna("Unknown")
    return df


def _convert_date_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy() # Prevent adverse effects
    target_columns_without_date: list = [
        'Purchase Order Description', 'Purchase Order (Contract) Number',
        'Revision Number', 'Specification Number', 'Contract Type',
        'Department', 'Vendor Name', 'Vendor ID', 'Address 1', 
        'Address 2', 'City', 'State', 'Zip',
        'Award Amount', 'Procurement Type', 'Contract PDF'
    ]
    remain_order_columns: list = df.columns.to_list()
    sql = f"""
        WITH raw_date_convertion AS (
            SELECT
                {', '.join(['"' + i + '"' for i in target_columns_without_date])},
                CASE
                WHEN 
                    LEFT(SPLIT_PART("Start Date", '/', 3), 1) = '0'
                THEN
                    SPLIT_PART("Start Date", '/', 1) || '/' || 
                    SPLIT_PART("Start Date", '/', 2) || '/' || 
                    '200' || RIGHT(SPLIT_PART("Start Date", '/', 3), 1)
                ELSE
                    "Start Date"
                END AS "Start Date",
                CASE
                WHEN
                    "End Date" = '11/30/3030'
                THEN
                    '12/31/2261'
                ELSE
                    "End Date"
                END AS "End Date",
                "Approval Date"
            FROM 
                df
        ), convert_type AS (
            SELECT
                {', '.join(['"' + i + '"' for i in target_columns_without_date])},
                STRPTIME("Start Date", '%m/%d/%Y') AS "Start Date",
                STRPTIME("End Date", '%m/%d/%Y') AS "End Date",
                STRPTIME("Approval Date", '%m/%d/%Y') AS "Approval Date"
            FROM
                raw_date_convertion
        )
        SELECT 
            * 
        FROM 
            convert_type
    """
    df = duckdb.sql(
        query=sql
    ).fetchdf()

    return df[remain_order_columns]


def _transform_data(df: pd.DataFrame):
    df = df.copy() # Prevent adverse effects
    sql: str = """
        SELECT
            -- Contract Duration (Days) = end_date - start_date (only for non-null values).
            DATE_DIFF('day', "Start Date", "End Date") AS "Contract Duration (Days)",
            
            --  Is Blanket Contract based on presence of end_date
            CASE
                WHEN 
                    "End Date" IS NULL
                THEN
                    TRUE
                ELSE
                    FALSE
            END AS "Is Blanket Contract",

            -- Flag records as Legacy Record if the contract number starts with an alphabet character.
            CASE
                WHEN 
                    REGEXP_MATCHES("Purchase Order (Contract) Number", '^[A-Za-z]')
                THEN 
                    TRUE
                ELSE
                    FALSE
            END AS "Legacy Record",

            -- Column Has Negative Modification where award_amount < 0.
            CASE
                WHEN
                    "Award Amount" < 0
                THEN
                    TRUE
                ELSE
                    FALSE
            END AS "Has Negative Modification"
        FROM
            df
    """
    df_extract_cols = duckdb.sql(
        query=sql
    ).fetchdf()

    return pd.concat([df, df_extract_cols], axis=1)


def main():
    FILE_PATH = "data/Competency Test_Contracts_20250721.csv"
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Process a file.")
    parser.add_argument(
        "--file", 
        type=str, 
        required=False, 
        help="Path to the input file",
        default=FILE_PATH
    )
    args: argparse.Namespace = parser.parse_args()
    conn: Optional[duckdb.DuckDBPyConnection] = None
    try:
        file_path: str = args.file
        df: pd.DataFrame = pd.read_csv(file_path)
        df = _handle_null_values(df=df)
        df = _convert_date_fields(df=df)
        df = _transform_data(df=df)

        conn = duckdb.connect()
        output_path: str = "part4.csv"
        conn.execute(f"COPY df TO '{output_path}' (HEADER, DELIMITER ',');")
        print(f"Saved transform data into {output_path}")
    except Exception as e:
        print(f"[Exception]: {e}")
    finally:
        if conn:
            conn.close()
            print("Closed duckdb connection!")


if __name__ == "__main__":
    main()