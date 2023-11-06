from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


class Gps1:
    def __init__(
        self,
        num_companies: int,
        num_years: int,
        mean_return: float,
        std_dev_return: float,
        seed: int = 22737928,
    ):
        np.random.seed(seed)

        self.num_companies = num_companies
        self.num_years = num_years
        self.mean_return = mean_return
        self.std_dev_return = std_dev_return

        # Generate random dataset for returns
        self.returns = np.random.normal(
            self.mean_return, std_dev_return, (self.num_companies, num_years)
        )
        self.df_returns = pd.DataFrame(
            self.returns, columns=[f"year_{i+1}" for i in range(num_years)]
        )

        self.ceo_female = np.random.choice([0, 1], size=self.num_companies)
        self.ceo_children = np.random.choice([0, 1, 2, 3], size=self.num_companies)
        self.ceo_age = np.random.uniform(40, 65, size=self.num_companies)
        self.num_employees = np.random.uniform(100, 10000, size=self.num_companies)
        self.equity_ratio = np.random.uniform(0, 1, size=self.num_companies)
        self.total_assets = np.random.uniform(10, 990, size=self.num_companies)
        self.employees_frac_male = np.random.uniform(0, 1, size=self.num_companies)
        self.employees_avg_age = np.random.normal(40, 12, size=self.num_companies)
        self.firm_sector = np.random.choice(
            range(1, 9), size=self.num_companies
        ) 

        self.df_firm_characteristics = pd.DataFrame(
            {
                "ceo_female": self.ceo_female,
                "ceo_children": self.ceo_children,
                "ceo_age": self.ceo_age,
                "num_employees": self.num_employees,
                "equity_ratio": self.equity_ratio,
                "total_assets": self.total_assets,
                "employees_frac_male": self.employees_frac_male,
                "employees_avg_age": self.employees_avg_age,
                "firm_sector": self.firm_sector,
            }
        )

        self.firm_characteristics = [
            "ceo_female",
            "ceo_children",
            "ceo_age",
            "num_employees",
            "equity_ratio",
            "total_assets",
            "employees_frac_male",
            "employees_avg_age",
        ]

        # Concatenate returns and firm characteristics DataFrames
        self.df_combined = pd.concat(
            [self.df_returns, self.df_firm_characteristics], axis=1
        )

    def run_regression_for_selected_sectors(
        self, df: pd.DataFrame, sectors: list, dependent_var: str, independent_vars: list
    ) -> sm.regression.linear_model.RegressionResultsWrapper:
        # Filter the DataFrame for the selected sectors
        sector_df = df[df["firm_sector"].isin(sectors)]
        X = sm.add_constant(sector_df[independent_vars])
        y = sector_df[dependent_var]
        model = sm.OLS(y, X).fit()
        return model

    def run_all_regressions(self, dependent_var: str, independent_vars: list) -> dict:
        model_dict = {}

        # All sectors
        all_sectors = list(range(1, 9))
        model = self.run_regression_for_selected_sectors(
            self.df_combined, all_sectors, dependent_var, independent_vars
        )
        model_dict[0] = model

        # Run OLS regressions for each sector
        for sector in range(1, 9):
            model = self.run_regression_for_selected_sectors(
                self.df_combined, [sector], dependent_var, independent_vars
            )
            model_dict[sector] = model

        return model_dict

    def find_statistically_significant_predictors(self, model: sm.regression.linear_model.RegressionResultsWrapper) -> pd.DataFrame:
        statistically_significant_predictors = pd.DataFrame()

        for sector, model in model.items():
            for predictor, pvalue in model.pvalues.items():
                if predictor != "const" and pvalue < 0.05:
                    significant_predictor = pd.DataFrame(
                        {
                            "predictor": predictor,
                            "pvalue": pvalue,
                            "coeff": model.params[predictor],
                        },
                        index=[sector],
                    )
                    statistically_significant_predictors = pd.concat(
                        [statistically_significant_predictors, significant_predictor],
                        axis=0,
                    )

        return statistically_significant_predictors

    def find_top_3_predictors(self, period: int) -> pd.DataFrame:
        year = f"year_{period}"

        models_without_interactions = self.run_all_regressions(
            year, self.firm_characteristics
        )

        # Create interactions between all pairs of firm characteristics, except for firm sector
        interaction_terms = []
        for combo in combinations(self.firm_characteristics, 2):
            interaction_name = f"{combo[0]}_X_{combo[1]}"
            self.df_combined[interaction_name] = (
                self.df_combined[combo[0]] * self.df_combined[combo[1]]
            )
            interaction_terms.append(interaction_name)

        all_vars = self.firm_characteristics + interaction_terms

        models_with_interactions = self.run_all_regressions(year, all_vars)

        statistically_significant_predictors_without_interactions = (
            self.find_statistically_significant_predictors(models_without_interactions)
        )
        statistically_significant_predictors_with_interactions = (
            self.find_statistically_significant_predictors(models_with_interactions)
        )

        statistically_significant_predictors = pd.concat(
            [
                statistically_significant_predictors_without_interactions,
                statistically_significant_predictors_with_interactions,
            ],
            axis=0,
        )
        # Uncomment if you want to see all predictors
        # print(statistically_significant_predictors)

        # Sort p-values and get top predictors
        top_3_predictors = statistically_significant_predictors.sort_values(
            by="pvalue"
        ).head(3)

        return top_3_predictors

    def plot_two_periods(
        self, df_fund_returns: pd.DataFrame, periods: list, superior_fund: str = None
    ):
        plt.figure(figsize=(11, 7))
        if superior_fund:
            df_fund_returns.loc[superior_fund, f"mean_return_t{periods[0]}"] += 0.01
            df_fund_returns.loc[superior_fund, f"mean_return_t{periods[1]}"] += 0.01

            fund_returns_to_plot_1 = df_fund_returns.loc[
                df_fund_returns.index != superior_fund, f"mean_return_t{periods[0]}"
            ]

            fund_returns_to_plot_2 = df_fund_returns.loc[
                df_fund_returns.index != superior_fund, f"mean_return_t{periods[1]}"
            ]

        else:
            fund_returns_to_plot_1 = df_fund_returns[f"mean_return_t{periods[0]}"]
            fund_returns_to_plot_2 = df_fund_returns[f"mean_return_t{periods[1]}"]

        # Plot for all original funds
        plt.scatter(
            fund_returns_to_plot_1,
            fund_returns_to_plot_2,
            color="blue",
            label="Mutual Funds",
        )

        if superior_fund:
            # Plot for the superior fund
            plt.scatter(
                df_fund_returns.loc["Fund 1", f"mean_return_t{periods[0]}"],
                df_fund_returns.loc["Fund 1", f"mean_return_t{periods[1]}"],
                color="orange",
                s=100,
                label="Superior Mutual Fund",
                edgecolors="black",
            )

        # Mean market return lines
        plt.axhline(
            y=self.mean_return,
            color="red",
            linestyle="--",
            label=f"Mean Market Return (t={periods[1]})",
        )
        plt.axvline(
            x=self.mean_return,
            color="green",
            linestyle="--",
            label=f"Mean Market Return (t={periods[0]})",
        )

        # Annotations and labels
        plt.title(f"Mutual Fund Performance in t={periods[0]} vs. t={periods[1]}")
        plt.xlabel(f"Mean Return in t={periods[0]}")
        plt.ylabel(f"Mean Return in t={periods[1]}")
        plt.legend()
        plt.grid(True)

        if superior_fund:
            plt.savefig(
                f"plots/mutual_fund_performance_t{periods[0]}_t{periods[1]}_superior.png"
            )
        else:
            plt.savefig(
                f"plots/mutual_fund_performance_t{periods[0]}_t{periods[1]}.png"
            )

    def plot_one_period(
        self, df_fund_returns: pd.DataFrame, period: int, funds_to_highlight: list = None
    ):
        plt.figure(figsize=(15, 9))
        # Plot sorted returns
        sorted_returns = df_fund_returns[f"mean_return_t{period}"].sort_values()
        sorted_returns.plot(
            kind="bar", color=(sorted_returns > 0).map({True: "g", False: "r"})
        )

        if funds_to_highlight:
            # Get the positions of the funds to highlight after sorting
            highlight_positions = [
                sorted_returns.index.get_loc(f"{num}") for num in funds_to_highlight
            ]

            # Highlight the bars for the specified funds
            for pos in highlight_positions:
                plt.gca().get_children()[pos].set_color("orange")

        # Add other plot elements and save the figure
        plt.axhline(y=0.07, color="blue", linestyle="--", label="Market Average Return")
        plt.title(f"Annual Returns for {len(sorted_returns)} Mutual Funds")
        plt.xlabel("Mutual Fund")
        plt.ylabel("Annual Return")
        plt.legend()

        plt.savefig(f"plots/mutual_fund_performance_t{period}.png")

    def plot_mutual_funds(
        self, periods: list, funds_to_highlight: list = None, superior_fund: str = None
    ) -> list:
        # Assign firms to mutual funds by slicing the DataFrame into chunks of 50 firms each
        funds = {
            f"Fund {i+1}": self.df_combined.iloc[i * 50 : (i + 1) * 50, :]
            for i in range(int(self.num_companies / 50))
        }

        # Calculate mean return for each fund in t=1 and t=2
        fund_returns = {
            fund_name: {
                "mean_return_t1": fund_data["year_1"].mean(),
                "mean_return_t2": fund_data["year_2"].mean(),
                "mean_return_t3": fund_data["year_3"].mean(),
            }
            for fund_name, fund_data in funds.items()
        }

        # Convert to DataFrame for easier plotting
        df_fund_returns = pd.DataFrame(fund_returns).T

        if len(periods) == 2:
            if superior_fund:
                self.plot_two_periods(df_fund_returns, periods, superior_fund)
            else:
                self.plot_two_periods(df_fund_returns, periods)

            # Identify any "star" fund that outperforms the market in both periods
            star_fund = df_fund_returns[
                (df_fund_returns[f"mean_return_t{periods[0]}"] > self.mean_return)
                & (df_fund_returns[f"mean_return_t{periods[1]}"] > self.mean_return)
            ]

            return star_fund.index.tolist()

        elif len(periods) == 1:
            self.plot_one_period(df_fund_returns, periods[0], funds_to_highlight)


def main():
    gps1 = Gps1(2000, 3, 0.07, 0.2)

    # Exercise a - Predicting Returns
    top_3_predictors_year_1 = gps1.find_top_3_predictors(period=1)
    print(top_3_predictors_year_1)
    print()

    # Exercise b, c - Mutual Fund Performance
    star_funds = gps1.plot_mutual_funds(periods=[1, 2], superior_fund="Fund 1")
    print(star_funds)
    print()

    # Exercise d - Are the findings robust?
    gps1.plot_mutual_funds(periods=[3], funds_to_highlight=star_funds)
    top_3_predictors_year_3 = gps1.find_top_3_predictors(period=3)

    print(top_3_predictors_year_3)

if __name__ == "__main__":
    main()
