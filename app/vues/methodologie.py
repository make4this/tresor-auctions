"""Page « Methodology » : question, sources, variables, validation, résultats et limites.

Les chiffres cités sont lus dans les fichiers du backtest et de la stratégie.
"""

from __future__ import annotations

import streamlit as st

from tresor import app_data as ad
from tresor import app_ui as ui
from tresor.formats import en

r2 = ad.backtest_r2(ui.csv("backtest_metrics.csv")).set_index(["family", "target"])
summary = ui.csv("strategy_summary.csv").set_index(["period", "rule"])
coupons = en(r2.loc[("nominal_coupon", "btc"), "r2_vs_mean3"] * 100, 1, sign=True)
loss_all = en(-summary.loc[("2014-2026", "strategie"), "mean_bp"], 1)
loss_recent = en(-summary.loc[("2020-2026", "strategie"), "mean_bp"], 1)

st.title("Methodology")
st.markdown(
    f"""
### The question
Before each US Treasury auction, how strong will demand be? Four measures are forecast:

- the **bid-to-cover ratio** (amount bid / amount sold);
- the **dealer share**: what the primary dealers, banks required to bid, end up buying;
- the **indirect share**: bidders such as foreign central banks and funds;
- the **tail**: the auction yield minus the market yield of the day before.

Notes and bonds (2 to 30 years) and bills (4 to 52 weeks) are modelled separately.

### The data, all public and free
- **US Treasury**: auction results since 2000 (Fiscal Data, TreasuryDirect), the auction
  calendar, the yield curve, bill rates, public debt and the Treasury's cash balance;
- **Cboe**: the VIX volatility index;
- **FRED** (St. Louis Fed): dollar index, Fed balance sheet, reverse repo, Baa credit spread,
  inflation expectations;
- **New York Fed**: primary dealer positions; **CFTC**: positioning in Treasury futures.

### Only what was known before the auction
Every value carries the date it became **public**, not the date it refers to. For an auction
on day *t*, a variable only uses values published **strictly before** *t*; a blocking check
rejects anything else. Where the archive of first releases exists (FRED's ALFRED), the real
first-release date is used.

### How the models are tested
- **Walk-forward**: every quarter since 2014, models are trained on all past auctions (since
  2009) and tested on the next quarter.
- **Benchmark**: the average of the last 3 auctions of the same maturity.
- **Reference forecast**: the average of 5 models (Ridge, Lasso, ElasticNet, Random Forest,
  LightGBM), chosen on principle rather than on results.
- **Pick and judge separately**: the best single model is picked on 2014-2019 and judged on
  2020-2026, with confidence intervals.

### Results
- The **bid-to-cover of notes and bonds** is the most reliable result: {coupons}% R² against
  the last-3 average over 2020-2026.
- The **extra sources** (dealers, CFTC, liquidity) improve no measure, and slightly hurt some.
- The **dealer share** of notes and bonds is poorly forecast since 2020; the **tail** mostly
  reflects the market move of the day.
- A **trading strategy** based on the signal loses {loss_all} bp per trade after costs over
  2014-2026 ({loss_recent} bp over 2020-2026) and does not beat always being short.

### Limits
- The Treasury announces auctions only a few days ahead, sometimes before their size (the size
  is then estimated).
- The tail is approximated with the previous close: the true benchmark, the 1 pm when-issued
  yield, is paid data.
- The 2015, 2021 and 2023 debt-ceiling episodes inflate part of the results for bills.
- The 0.5 bp cost per trade is an assumption.

### Glossary
- **bp**: basis point, 0.01 percentage point of yield.
- **R² against the last-3 average**: 1 − (model squared error / squared error of the last-3
  average); above 0, the model does better.
- **Duration**: sensitivity of a bond to yields; *short duration* gains when yields rise.
- **Permutation / SHAP**: what a variable adds out of sample / how much it weighs in the model.
"""
)
