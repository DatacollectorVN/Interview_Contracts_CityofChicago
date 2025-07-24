import streamlit as st
import pandas as pd
import plotly.express as px
import duckdb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

LIMIT_RENDER: int = 100 # Just limit the number of contract could be rendered.


@st.cache_data
def _read_data():
    return pd.read_csv("part4.csv") 


@st.cache_data
def _get_amendment_count_df(df: pd.DataFrame):
    # Run SQL to count amendments per contract
    sql: str = """
    WITH amendment_count AS (
    SELECT
        "Purchase Order (Contract) Number",
        COUNT(*) AS amendment_count,
        SUM("Award Amount") AS total_award_amount
    FROM
        df
    GROUP BY
        "Purchase Order (Contract) Number"
    )
    SELECT
        "Purchase Order (Contract) Number",
        CASE
            WHEN
                amendment_count < 50
            THEN
                'Low'
            WHEN
                amendment_count >= 50
                AND amendment_count < 100
            THEN
                'Moderate'
            ELSE
                'High'
        END AS "risk",
        amendment_count,
        total_award_amount
    FROM
        amendment_count
    ORDER BY
        amendment_count DESC
    """
    return duckdb.query(sql).fetchdf()


@st.cache_data
def _get_trend_overtime_df(df: pd.DataFrame):
    sql: str = """
        WITH daily_calculation AS (
            SELECT
                "Start Date"::DATE AS "start_date_daily",
                SUM("Award Amount") AS "sum_award_daily",
            FROM
                df
            WHERE
                "Award Amount" IS NOT NULL
                AND "Start Date" IS NOT NULL 
            GROUP BY 
                start_date_daily
        )
        SELECT
            *,
            -- Monthly calculation
            DATE_TRUNC('month', start_date_daily) AS start_date_monthly,
            SUM(sum_award_daily) OVER (
                PARTITION BY DATE_TRUNC('month', "start_date_daily")
            ) AS "sum_award_monthly",
            
            -- Yearly calculation
            DATE_TRUNC('year', start_date_daily) AS start_date_yearly,
            SUM(sum_award_daily) OVER (
                PARTITION BY DATE_TRUNC('year', "start_date_daily")
            ) AS "sum_award_yearly"
        FROM
            daily_calculation
    """
    return duckdb.query(sql).fetchdf()


@st.cache_data
def _get_department_spend(df: pd.DataFrame):
    sql: str = """
        WITH raw_compute AS (
            SELECT
                "Purchase Order (Contract) Number",
                "Department",
                DATE_DIFF('day', MIN("Start Date"::DATE), MAX("End Date"::DATE)) AS contract_duration_day,
                SUM("Award Amount") AS "sum_award",
            FROM
                df
            GROUP BY
                "Purchase Order (Contract) Number",
                "Department"
            HAVING
                contract_duration_day IS NOT NULL
        )
        SELECT
            "Department",
            SUM(sum_award) AS sum_award,
            AVG(contract_duration_day) AS avg_contract_duration_day,
            SUM(contract_duration_day) AS sum_contract_duration_day,
            COUNT(DISTINCT "Purchase Order (Contract) Number") AS number_of_contract,
            MIN(contract_duration_day) AS shortest_duration,
            MAX(contract_duration_day) AS longest_duration,
        FROM
            raw_compute
        GROUP BY
            "Department"
        ORDER BY
            sum_award DESC
    """
    df = duckdb.query(sql).fetchdf()
    model = LinearRegression()
    model.fit(df[["sum_award"]], df["sum_contract_duration_day"])
    df["regression_line"] = model.predict(df[["sum_award"]])
    return df


@st.cache_data
def _get_vendor_spend(df: pd.DataFrame):
    sql: str = """
        SELECT
            "Vendor Name" AS vendor,
            COUNT(DISTINCT "Purchase Order (Contract) Number") AS number_of_contracts,
            SUM("Award Amount") AS total_award_amount,
            AVG("Award Amount") AS avg_award_amount
        FROM df
        WHERE 
            "Vendor Name" IS NOT NULL
            AND "Award Amount" IS NOT NULL
        GROUP BY 
            "Vendor Name"
        ORDER BY 
            total_award_amount DESC
    """
    return duckdb.query(sql).fetchdf()
    

class CacheManager:
    dataset: dict = {}


def _prepare_data():
    # In that case, I want to cache all the transformed data frames in memory (@st.cache_data) to speed up performance at the expense of memory space.
    CacheManager.dataset["df"] = _read_data()
    CacheManager.dataset["df_amendment"] = _get_amendment_count_df(df=CacheManager.dataset["df"])
    CacheManager.dataset["df_trend_overtime"] = _get_trend_overtime_df(df=CacheManager.dataset["df"])
    CacheManager.dataset["df_department_spend"] = _get_department_spend(df=CacheManager.dataset["df"])
    CacheManager.dataset["df_vendor_spend"] = _get_vendor_spend(df=CacheManager.dataset["df"])


def _render_amendment_chart():
    is_limit: bool = False

    st.subheader("Amendment Count by Contract Number")
    df: pd.DataFrame = CacheManager.dataset["df_amendment"]
    contract_options: list = sorted(df["Purchase Order (Contract) Number"].dropna().unique().tolist())
    selected_contracts = st.multiselect(
        "Select Contract Numbers",
        options=["All"] + contract_options,
        default=["All"]
    )
    if "All" in selected_contracts:
        is_limit = True
        selected_contracts = contract_options

    risk_options: list = sorted(df["risk"].dropna().unique().tolist())
    selected_risks = st.multiselect(
        "Select Risk Levels",
        options=["All"] + risk_options,
        default=["All"]
    )

    if "All" in selected_risks:
        selected_risks = risk_options

    # Combined filter
    filtered_df = df[
        df["Purchase Order (Contract) Number"].isin(selected_contracts) &
        df["risk"].isin(selected_risks)
    ]

    if is_limit:
        st.warning("Only show top 100!")
        filtered_df = filtered_df.iloc[:LIMIT_RENDER, :]

    risk_color_map = {"High": "red", "Moderate": "yellow", "Low": "blue"}

    if not filtered_df.empty:
        fig = px.bar(
            filtered_df,
            x="Purchase Order (Contract) Number",
            y="amendment_count",
            color="risk",
            color_discrete_map=risk_color_map,
            title="Amendment Count by Contract Number (Colored by Risk)",
            hover_data={"total_award_amount": True, "amendment_count": True, "risk": True}
        )
        fig.update_layout(xaxis_title="Contract Number", yaxis_title="Amendment Count", xaxis_type="category")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data matches your selected filters.")


def _render_trend_overtime_chart():
    st.subheader("Award Amount")
    df: pd.DataFrame = CacheManager.dataset["df_trend_overtime"]

    min_date = df["start_date_daily"].min()
    max_date = df["start_date_daily"].max()
    try:
        start_date, end_date = st.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    except Exception as e:
        st.warning("Need to select date range!")
        return
    
    filtered_df = df[(df["start_date_daily"] >= pd.to_datetime(start_date)) & (df["start_date_daily"] <= pd.to_datetime(end_date))]

    group_selection = st.selectbox(
        "Select Group",
        ("daily", "monthly", "yearly"),
        index=0,
    )
    
    fig = px.bar(
        filtered_df,
        x=f"start_date_{group_selection}",
        y=f"sum_award_{group_selection}",
        title="Total Award Amount by Contract Start Date",
        labels={f"start_date_{group_selection}": "Contract Start Date", f"sum_award_{group_selection}": "Total Award ($)"},
        hover_data={f"sum_award_{group_selection}": ":,.0f"}
    )

    fig.update_layout(
        xaxis_title="Start Date",
        yaxis_title="Total Award Amount ($)",
        xaxis_tickformat="%Y-%m-%d",
        height=800
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_department_spend():
    st.subheader("Depearment Analysis")
    df: pd.DataFrame = CacheManager.dataset["df_department_spend"]

    fig1 = px.bar(
        df,
        x="sum_award",
        y="Department",     
        orientation="h",
        title="Total Award Amount by Department",
        hover_data={
            "avg_contract_duration_day": ":,.0f",
            "sum_contract_duration_day": ":,.0f",
            "number_of_contract": ":,.0f", 
            "shortest_duration": True, "longest_duration": True,
        }
    )

    fig1.update_layout(
        xaxis_title="Total Award Amount ($)",
        yaxis_title="Department",
        height=1000
    )
    
    st.plotly_chart(fig1, use_container_width=True)

    # Calculate correlation
    correlation = df["sum_contract_duration_day"].corr(df["sum_award"])
    st.markdown(f"### 📈 Correlation (Pearson): `{correlation:.4f}`")

    fig2 = px.scatter(df, x="sum_award", y="sum_contract_duration_day", title="Scatter Plot with Regression Line")
    fig2.add_scatter(x=df["sum_award"], y=df["regression_line"], mode="lines", name="Regression Line", showlegend=False)

    fig2.update_layout(xaxis_title="Total Award Amount ($)", yaxis_title="Contract duration (day)")
    st.plotly_chart(fig2, use_container_width=True)


def _render_vendor_spend():
    st.subheader("Vendor Analysis")
    df: pd.DataFrame = CacheManager.dataset["df_vendor_spend"]

    vendor_options: list = df["vendor"].unique().tolist()
    selected_vendors = st.multiselect(
        "Select Vendor",
        options=["All"] + vendor_options,
        default=["All"]
    )
    
    if not selected_vendors:
        st.warning("Need to select vendor!")
        return

    filtered_df: pd.DataFrame
    if "All" in selected_vendors:
        st.warning("Only show top 100!")
        filtered_df = df.iloc[:LIMIT_RENDER, :]
    else:
        filtered_df = df[df["vendor"].isin(selected_vendors)]

    metric_selection = st.selectbox(
        "Select Metric",
        ("number_of_contracts", "total_award_amount", "avg_award_amount"),
        index=0,
    )

    metric_map: dict = {
        "number_of_contracts": "Total contracts",
        "total_award_amount": "Total Award Amount ($)",
        "avg_award_amount": "Average Award Amount ($)",
    }

    filtered_df = filtered_df.sort_values(by=metric_selection, ascending=False)

    fig = px.bar(
        filtered_df,
        x=metric_selection,
        y="vendor",     
        orientation="h",
        title="Vendor Analysis",
        hover_data={f"number_of_contracts": ":,.0f", 
            "total_award_amount": ":,.0f", 
            "avg_award_amount": ":,.0f",
        }
    )

    fig.update_layout(
        xaxis_title=metric_map[metric_selection],
        yaxis_title="vendor",
        height=1000
    )
    
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.set_page_config(
        page_title="Chicago Contracts Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    _prepare_data()
    st.title("Chicago Vendor Contracts Dashboard")
    _render_amendment_chart()
    _render_trend_overtime_chart()
    _render_department_spend()
    _render_vendor_spend()
    

if __name__ == "__main__":
    main()

    