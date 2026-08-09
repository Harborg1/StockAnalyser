import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#https://www.sec.gov/edgar/search/#/dateRange=10y&category=form-cat1&ciks=0001652044&entityName=Alphabet%2520Inc.%2520(GOOG%252C%2520GOOGL)%2520(CIK%25200001652044)


# =========================
# CONFIG
# =========================

TICKER_CIKS = {
    "TSLA": "0001318605",
    "NVDA": "0001045810"
}

SEC_USER_AGENT = "StockAnalyser thesis project s214967@dtu.dk"

SAVE_CSV = True


# =========================
# SEC REQUEST
# =========================

def get_sec_companyfacts(cik):
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

    headers = {
        "User-Agent": SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov"
    }

    response = requests.get(url, headers=headers, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(
            f"SEC request failed for CIK {cik}. "
            f"Status code: {response.status_code}. "
            f"Response: {response.text[:300]}"
        )

    time.sleep(0.2)

    return response.json()


# =========================
# DEBUG TAG SEARCH
# =========================

def inspect_sec_tags(cik, keyword):
    data = get_sec_companyfacts(cik)

    facts = data.get("facts", {}).get("us-gaap", {})

    matches = []

    for tag in facts.keys():
        if keyword.lower() in tag.lower():
            matches.append(tag)

    print(f"\nTags for CIK {cik} containing '{keyword}':")

    for tag in matches:
        print(tag)

    return matches


# =========================
# RAW SEC FACT EXTRACTION
# =========================

def get_sec_fact_rows_for_tag(data, tag, unit="USD"):
    facts = data.get("facts", {}).get("us-gaap", {})

    if tag not in facts:
        return pd.DataFrame()

    units = facts[tag].get("units", {})

    if unit not in units:
        return pd.DataFrame()

    rows = pd.DataFrame(units[unit])

    if rows.empty:
        return pd.DataFrame()

    keep_cols = [
        "fy",
        "fp",
        "form",
        "filed",
        "start",
        "end",
        "val",
        "accn",
        "frame"
    ]

    rows = rows[[col for col in keep_cols if col in rows.columns]].copy()

    rows["filed"] = pd.to_datetime(rows["filed"], errors="coerce")
    rows["end"] = pd.to_datetime(rows["end"], errors="coerce")

    if "start" in rows.columns:
        rows["start"] = pd.to_datetime(rows["start"], errors="coerce")
        rows["period_days"] = (rows["end"] - rows["start"]).dt.days
    else:
        rows["period_days"] = np.nan

    rows["tag"] = tag
    rows["value"] = pd.to_numeric(rows["val"], errors="coerce")

    rows = rows.dropna(subset=["filed", "end", "value"])

    return rows


# =========================
# FLOW FACT CLEANING
# =========================

def build_quarterly_flow_rows(rows):
    """
    Flow metrics are income statement / cash flow metrics.

    10-Q rows with period_days around 90 are used directly.

    10-K annual rows are not used directly because they are annual totals.
    Instead, Q4 is estimated as:

        FY annual value - Q1 - Q2 - Q3
    """

    if rows.empty:
        return pd.DataFrame()

    rows = rows.copy()

    rows = rows[rows["form"].isin(["10-Q", "10-K"])]

    if rows.empty:
        return pd.DataFrame()

    quarterly = rows[
        (rows["form"] == "10-Q") &
        (rows["period_days"].between(70, 120))
    ].copy()

    annual = rows[
        (rows["form"] == "10-K") &
        (rows["period_days"].between(300, 400))
    ].copy()

    if not quarterly.empty:
        quarterly = quarterly.sort_values(["end", "filed"])
        quarterly = quarterly.drop_duplicates(subset=["end"], keep="first")

    q4_rows = []

    if not annual.empty and not quarterly.empty:
        annual = annual.sort_values(["end", "filed"])
        annual = annual.drop_duplicates(subset=["end"], keep="first")

        for _, annual_row in annual.iterrows():
            fy = annual_row.get("fy", np.nan)

            q1_q3 = quarterly[
                (quarterly["fy"] == fy) &
                (quarterly["fp"].isin(["Q1", "Q2", "Q3"]))
            ]

            if len(q1_q3) == 3:
                q4_value = annual_row["value"] - q1_q3["value"].sum()

                q4_row = annual_row.copy()
                q4_row["value"] = q4_value
                q4_row["form"] = "10-K_DERIVED_Q4"
                q4_row["fp"] = "Q4"
                q4_row["period_days"] = np.nan

                q4_rows.append(q4_row)

    if len(q4_rows) > 0:
        q4 = pd.DataFrame(q4_rows)
        out = pd.concat([quarterly, q4], axis=0)
    else:
        out = quarterly

    if out.empty:
        return pd.DataFrame()

    out = out.sort_values(["end", "filed"])

    out = out[[
        "filed",
        "end",
        "value",
        "form",
        "fp",
        "tag"
    ]].copy()

    return out


# =========================
# INSTANT FACT CLEANING
# =========================

def build_instant_rows(rows):
    """
    Instant metrics are balance sheet metrics.

    These are point-in-time values, so 10-Q and 10-K values are valid.
    """

    if rows.empty:
        return pd.DataFrame()

    rows = rows.copy()

    rows = rows[rows["form"].isin(["10-Q", "10-K"])]

    if rows.empty:
        return pd.DataFrame()

    rows = rows.sort_values(["end", "filed"])
    rows = rows.drop_duplicates(subset=["end"], keep="first")

    rows = rows[[
        "filed",
        "end",
        "value",
        "form",
        "fp",
        "tag"
    ]].copy()

    return rows


# =========================
# METRIC BUILDER
# =========================

def sec_metric(data, name, tag_candidates, unit="USD", fact_type="flow"):
    """
    Tries every tag candidate.

    This fixes the earlier issue where the first existing tag could be found,
    but it had no usable quarterly rows.
    """

    for tag in tag_candidates:
        raw_rows = get_sec_fact_rows_for_tag(
            data=data,
            tag=tag,
            unit=unit
        )

        if fact_type == "flow":
            cleaned_rows = build_quarterly_flow_rows(raw_rows)
        elif fact_type == "instant":
            cleaned_rows = build_instant_rows(raw_rows)
        else:
            raise ValueError("fact_type must be either 'flow' or 'instant'")

        if cleaned_rows.empty:
            continue

        out = cleaned_rows[[
            "filed",
            "end",
            "value",
            "form",
            "fp",
            "tag"
        ]].copy()

        out = out.rename(columns={
            "filed": "Filing_Date",
            "end": f"{name}_Period_End",
            "value": name,
            "form": f"{name}_Form",
            "fp": f"{name}_Fiscal_Period",
            "tag": f"{name}_Tag"
        })

        out = out.set_index("Filing_Date")
        out = out.sort_index()

        out = out[~out.index.duplicated(keep="first")]

        print(f"{name}: using SEC tag {tag}")

        return out

    print(f"{name}: no usable SEC tag found")

    return pd.DataFrame(columns=[
        name,
        f"{name}_Period_End",
        f"{name}_Form",
        f"{name}_Fiscal_Period",
        f"{name}_Tag"
    ])


# =========================
# BUILD COMPANY FUNDAMENTALS
# =========================

def build_company_fundamentals(ticker, cik):
    print(f"\nFetching SEC fundamentals for {ticker}")

    data = get_sec_companyfacts(cik)

    print(f"Company name: {data.get('entityName')}")

    metrics = []

    metrics.append(sec_metric(
        data=data,
        name="Revenue",
        tag_candidates=[
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet"
        ],
        unit="USD",
        fact_type="flow"
    ))

    metrics.append(sec_metric(
        data=data,
        name="Cost_Of_Revenue",
        tag_candidates=[
            "CostOfRevenue",
            "CostOfGoodsAndServicesSold",
            "CostOfGoodsSold"
        ],
        unit="USD",
        fact_type="flow"
    ))

    metrics.append(sec_metric(
        data=data,
        name="Gross_Profit",
        tag_candidates=[
            "GrossProfit"
        ],
        unit="USD",
        fact_type="flow"
    ))

    metrics.append(sec_metric(
        data=data,
        name="Operating_Income",
        tag_candidates=[
            "OperatingIncomeLoss"
        ],
        unit="USD",
        fact_type="flow"
    ))

    metrics.append(sec_metric(
        data=data,
        name="Net_Income",
        tag_candidates=[
            "NetIncomeLoss",
            "ProfitLoss"
        ],
        unit="USD",
        fact_type="flow"
    ))

    metrics.append(sec_metric(
        data=data,
        name="Operating_Cash_Flow",
        tag_candidates=[
            "NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"
        ],
        unit="USD",
        fact_type="flow"
    ))

    metrics.append(sec_metric(
        data=data,
        name="Capex",
        tag_candidates=[
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PaymentsToAcquireProductiveAssets"
        ],
        unit="USD",
        fact_type="flow"
    ))

    metrics.append(sec_metric(
        data=data,
        name="Assets",
        tag_candidates=[
            "Assets"
        ],
        unit="USD",
        fact_type="instant"
    ))

    metrics.append(sec_metric(
        data=data,
        name="Liabilities",
        tag_candidates=[
            "Liabilities",
            "LiabilitiesCurrent"
        ],
        unit="USD",
        fact_type="instant"
    ))

    metrics.append(sec_metric(
        data=data,
        name="Equity",
        tag_candidates=[
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"
        ],
        unit="USD",
        fact_type="instant"
    ))

    metrics.append(sec_metric(
        data=data,
        name="Cash",
        tag_candidates=[
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"
        ],
        unit="USD",
        fact_type="instant"
    ))

    metrics.append(sec_metric(
        data=data,
        name="Long_Term_Debt",
        tag_candidates=[
            "LongTermDebt",
            "LongTermDebtNoncurrent",
            "LongTermDebtAndFinanceLeaseObligationsNoncurrent"
        ],
        unit="USD",
        fact_type="instant"
    ))

    fundamentals = pd.concat(metrics, axis=1)

    fundamentals = fundamentals.sort_index()
    fundamentals = fundamentals[~fundamentals.index.duplicated(keep="first")]

    fundamentals["Ticker"] = ticker
    fundamentals["CIK"] = cik

    period_cols = [
        col for col in fundamentals.columns
        if col.endswith("_Period_End")
    ]

    if len(period_cols) > 0:
        fundamentals["Period_End"] = fundamentals[period_cols].bfill(axis=1).iloc[:, 0]
        fundamentals["Period_End"] = pd.to_datetime(
            fundamentals["Period_End"],
            errors="coerce"
        )

    fundamentals = fundamentals.replace([np.inf, -np.inf], np.nan)

    fundamentals = add_derived_metrics(fundamentals)

    return fundamentals


# =========================
# DERIVED METRICS
# =========================

def add_derived_metrics(df):
    df = df.copy()

    df = df.sort_index()

    if "Revenue" in df.columns:
        df["Revenue_Growth_QoQ"] = df["Revenue"].pct_change(fill_method=None)
        df["Revenue_Growth_YoY"] = df["Revenue"].pct_change(4, fill_method=None)

    if "Net_Income" in df.columns:
        df["Net_Income_Growth_QoQ"] = df["Net_Income"].pct_change(fill_method=None)
        df["Net_Income_Growth_YoY"] = df["Net_Income"].pct_change(4, fill_method=None)

    if "Gross_Profit" in df.columns and "Revenue" in df.columns:
        df["Gross_Margin"] = df["Gross_Profit"] / df["Revenue"]

    if "Operating_Income" in df.columns and "Revenue" in df.columns:
        df["Operating_Margin"] = df["Operating_Income"] / df["Revenue"]

    if "Net_Income" in df.columns and "Revenue" in df.columns:
        df["Profit_Margin"] = df["Net_Income"] / df["Revenue"]

    if "Long_Term_Debt" in df.columns and "Assets" in df.columns:
        df["Debt_To_Assets"] = df["Long_Term_Debt"] / df["Assets"]

    if "Cash" in df.columns and "Assets" in df.columns:
        df["Cash_To_Assets"] = df["Cash"] / df["Assets"]

    if "Liabilities" in df.columns and "Assets" in df.columns:
        df["Liabilities_To_Assets"] = df["Liabilities"] / df["Assets"]

    if "Operating_Cash_Flow" in df.columns and "Capex" in df.columns:
        df["Free_Cash_Flow"] = df["Operating_Cash_Flow"] - df["Capex"].abs()

    if "Free_Cash_Flow" in df.columns and "Revenue" in df.columns:
        df["FCF_Margin"] = df["Free_Cash_Flow"] / df["Revenue"]

    df = df.replace([np.inf, -np.inf], np.nan)

    return df


# =========================
# BUILD DATASET
# =========================

def build_company_dataset():
    all_data = []

    for ticker, cik in TICKER_CIKS.items():
        fundamentals = build_company_fundamentals(
            ticker=ticker,
            cik=cik
        )

        all_data.append(fundamentals)

        print(f"\n{ticker} fundamentals preview")
        print(fundamentals.tail(10))

        if SAVE_CSV:
            filename = f"{ticker}_sec_fundamentals.csv"
            fundamentals.to_csv(filename)
            print(f"Saved {filename}")

    combined = pd.concat(all_data, axis=0)
    combined = combined.sort_index()

    if SAVE_CSV:
        combined.to_csv("combined_sec_fundamentals_tsla_nvda.csv")
        print("Saved combined_sec_fundamentals_tsla_nvda.csv")

    return combined


# =========================
# QUARTERLY COMPARISON TABLE
# =========================

def make_quarterly_comparison(combined):
    df = combined.copy()

    df = df.reset_index()

    if "Filing_Date" not in df.columns:
        df = df.rename(columns={"index": "Filing_Date"})

    df["Filing_Date"] = pd.to_datetime(df["Filing_Date"], errors="coerce")

    if "Period_End" in df.columns:
        df["Quarter_End"] = pd.to_datetime(df["Period_End"], errors="coerce")
    else:
        df["Quarter_End"] = df["Filing_Date"]

    df["Quarter_Available_Date"] = df["Filing_Date"]

    comparison_cols = [
        "Ticker",
        "CIK",
        "Quarter_Available_Date",
        "Quarter_End",
        "Revenue",
        "Revenue_Growth_QoQ",
        "Revenue_Growth_YoY",
        "Cost_Of_Revenue",
        "Gross_Profit",
        "Gross_Margin",
        "Operating_Income",
        "Operating_Margin",
        "Net_Income",
        "Profit_Margin",
        "Net_Income_Growth_QoQ",
        "Net_Income_Growth_YoY",
        "Operating_Cash_Flow",
        "Capex",
        "Free_Cash_Flow",
        "FCF_Margin",
        "Assets",
        "Liabilities",
        "Equity",
        "Cash",
        "Long_Term_Debt",
        "Debt_To_Assets",
        "Cash_To_Assets",
        "Liabilities_To_Assets"
    ]

    comparison_cols = [
        col for col in comparison_cols
        if col in df.columns
    ]

    comparison = df[comparison_cols].copy()

    comparison = comparison.sort_values([
        "Quarter_End",
        "Ticker",
        "Quarter_Available_Date"
    ])

    return comparison


# =========================
# PLOTS
# =========================

def plot_metric(comparison, metric):
    if metric not in comparison.columns:
        print(f"{metric} not found.")
        return

    plt.figure(figsize=(12, 6))

    for ticker in sorted(comparison["Ticker"].dropna().unique()):
        data = comparison[comparison["Ticker"] == ticker].copy()
        data = data.dropna(subset=["Quarter_End", metric])

        if data.empty:
            continue

        plt.plot(
            data["Quarter_End"],
            data[metric],
            marker="o",
            label=ticker
        )

    plt.title(f"{metric} by Quarter")
    plt.xlabel("Quarter")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_main_metrics(comparison):
    metrics = [
        "Revenue",
        "Revenue_Growth_YoY",
        "Gross_Margin",
        "Operating_Margin",
        "Profit_Margin",
        "Free_Cash_Flow",
        "FCF_Margin",
        "Debt_To_Assets",
        "Cash_To_Assets",
        "Liabilities_To_Assets"
    ]

    for metric in metrics:
        plot_metric(comparison, metric)


# =========================
# DIAGNOSTICS
# =========================

def print_missing_summary(comparison):
    print("\nMissing values by column")
    print(comparison.isna().sum().sort_values(ascending=False))


def print_latest_comparison(comparison, n=30):
    print("\nQuarterly comparison")
    print(comparison.tail(n).to_string(index=False))


def print_metric_coverage(comparison):
    print("\nMetric coverage by ticker")

    metric_cols = [
        col for col in comparison.columns
        if col not in [
            "Ticker",
            "CIK",
            "Quarter_Available_Date",
            "Quarter_End"
        ]
    ]

    coverage = (
        comparison
        .groupby("Ticker")[metric_cols]
        .apply(lambda x: x.notna().sum())
    )

    print(coverage)


# =========================
# MAIN
# =========================

def main():
    combined = build_company_dataset()

    comparison = make_quarterly_comparison(combined)

    print_latest_comparison(comparison, n=40)

    print_missing_summary(comparison)

    print_metric_coverage(comparison)

    if SAVE_CSV:
        comparison.to_csv(
            "quarterly_company_comparison_tsla_nvda.csv",
            index=False
        )

        print("Saved quarterly_company_comparison_tsla_nvda.csv")

    plot_main_metrics(comparison)

    return combined, comparison


if __name__ == "__main__":
    combined, comparison = main()