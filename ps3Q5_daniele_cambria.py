import pandas as pd
import numpy as np

"""
pandas==2.1.10
numpy==1.25.2


Python 3.10.8

Notes:
As some ratios are calculated as, e.g., 100*(P1/P2), in excel the ratio I extracted is
adjusted to just P1/P2. This is done to avoid having to divide by 100 in the code just
in some cases. 

As the returns from the companies are arithmetic returns, and the S&P500 returns are
log returns, I also converted the company returns to log returns for comparability.

The companies seem to generally have lower returns than the S&P500, but also lower risk.
I would have liked to adjust for risk using the CAPM, but I don't know how I could do it.
"""


def extract_data_from_excel(file_path):
    xls = pd.ExcelFile(file_path)

    sheet_names = xls.sheet_names

    data_dict = {}
    for sheet in sheet_names:
        data_dict[sheet] = pd.read_excel(file_path, sheet_name=sheet)

    return data_dict


def determine_investment_strategy(df):
    # Create columns for the weights of P1 and P2
    df["Weight P1"] = 0.0
    df["Weight P2"] = 0.0

    # Loop through the rows of the DataFrame
    for i in range(len(df)):
        actual_ratio = df.loc[i, "True Ratio"]
        theoretical_ratio = df.loc[i, "Ratio"]

        # If the actual ratio is 10% or more than the theoretical ratio
        if actual_ratio >= 1.10 * theoretical_ratio:
            df.loc[i, "Weight P1"] = -1
            df.loc[i, "Weight P2"] = actual_ratio
        # If the actual ratio is 10% or less than the theoretical ratio
        elif actual_ratio <= 0.90 * theoretical_ratio:
            df.loc[i, "Weight P1"] = 1 / actual_ratio
            df.loc[i, "Weight P2"] = -1

    return df


def investment_earnings(df):
    # Initialize previous weights to None (indicating no position yet)
    prev_weight_p1 = None
    prev_weight_p2 = None

    # Initialize columns to store daily earnings, previous weights, and total daily earnings
    df["Earnings P1"] = 0.0
    df["Earnings P2"] = 0.0
    df["Total Daily Earnings"] = 0.0
    df["prev_weight_p1"] = 0.0
    df["prev_weight_p2"] = 0.0

    # Loop through the rows of the DataFrame
    for i in range(len(df)):
        # If a position was previously established, maintain it
        if prev_weight_p1 is not None:
            # Update the previous weights, so that the next position can use the same values
            df.loc[i, "prev_weight_p1"] = prev_weight_p1
            df.loc[i, "prev_weight_p2"] = prev_weight_p2

            # Calculate the earnings for each position, using the weights from the first day of the position
            df.loc[i, "Earnings P1"] = (
                df.loc[i, "P1"] - df.loc[i - 1, "P1"]
            ) * prev_weight_p1
            df.loc[i, "Earnings P2"] = (
                df.loc[i, "P2"] - df.loc[i - 1, "P2"]
            ) * prev_weight_p2

            df.loc[i, "Total Daily Earnings"] = (
                df.loc[i, "Earnings P1"] + df.loc[i, "Earnings P2"]
            )

        # Update the previous weights if a new position is established
        if (
            df.loc[i, "Weight P1"] != prev_weight_p1
            and df.loc[i, "Weight P2"] != prev_weight_p2
        ):
            prev_weight_p1 = df.loc[i, "Weight P1"]
            prev_weight_p2 = df.loc[i, "Weight P2"]

    # Calculate cumulative earnings
    total_earnings = df["Total Daily Earnings"].sum()

    return df, total_earnings


file_path = "data/ps3q5_data.xlsx"
data = extract_data_from_excel(file_path)

earnings_dict = {}
non_zero_weights_rows = {}
average_returns = {}
risks = {}
log_average_returns = {}
log_risks = {}
sp500_avg_return = {}
sp500_risk = {}

for comp in data:
    df = data[comp]
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    df["True Ratio"] = df["P1"] / df["P2"]

    data[comp] = determine_investment_strategy(df)
    data[comp], earnings_dict[comp] = investment_earnings(data[comp])

    print(f"Total earnings for {comp}: {earnings_dict[comp]}")

    # Calculate daily returns for the strategy
    df["Strategy Daily Returns"] = df["Total Daily Earnings"] / (
        abs(df["P1"]) + abs(df["P2"])
    )

    # As the S&P500 returns are log returns, also compute the log returns for the companies, for comparability
    df["Log returns"] = np.log(1 + df["Strategy Daily Returns"])

    filename = f"data/{comp}_inv.csv"
    df.to_csv(filename, index=False)

    # Compute average return and risk (standard deviation) for the strategy
    average_returns[comp] = df["Strategy Daily Returns"].mean()
    risks[comp] = df["Strategy Daily Returns"].std()

    log_average_returns[comp] = df["Log returns"].mean()
    log_risks[comp] = df["Log returns"].std()

    # Compute average return and risk (standard deviation) for the S&P 500
    sp500_avg_return[comp] = df["Reg"].mean()
    sp500_risk[comp] = df["Reg"].std()

    print(f"Average return for {comp}: {average_returns[comp]}")
    print(f"Risk for {comp}: {risks[comp]}")
    print(f"Log Average return for {comp}: {log_average_returns[comp]}")
    print(f"Log Risk for {comp}: {log_risks[comp]}")
    print(f"Log average return for S&P 500: {sp500_avg_return[comp]}")
    print(f"Log risk for S&P 500: {sp500_risk[comp]}")
    print()

# Calculate the overall average return and risk across all companies
overall_avg_return = sum(average_returns.values()) / len(average_returns)
overall_risk = sum(risks.values()) / len(risks)
overall_log_avg_return = sum(log_average_returns.values()) / len(log_average_returns)
overall_log_risk = sum(log_risks.values()) / len(log_risks)
overall_sp500_avg_return = sum(sp500_avg_return.values()) / len(sp500_avg_return)
overall_sp500_risk = sum(sp500_risk.values()) / len(sp500_risk)

print(f"Overall average return: {overall_avg_return}")
print(f"Overall risk: {overall_risk}")
print(f"Overall log average return: {overall_log_avg_return}")
print(f"Overall log risk: {overall_log_risk}")
print(f"Overall log average return for S&P 500: {overall_sp500_avg_return}")
print(f"Overall log risk for S&P 500: {overall_sp500_risk}")
