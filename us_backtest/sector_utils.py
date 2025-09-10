import pandas as pd
import yfinance as yf


def get_sp500_constituents() -> pd.DataFrame:
    """Fetch S&P500 constituents with sector info.

    Primary source: Wikipedia. Fallback: yfinance (sector field), where sub-industry may be missing.
    """
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0].copy()
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
        return df[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]]
    except Exception:
        # Fallback: use yfinance to get list and sector metadata (best-effort)
        syms = yf.tickers_sp500()
        recs = []
        for s in syms:
            try:
                info = yf.Ticker(s).get_info()
                recs.append({
                    "Symbol": s,
                    "Security": info.get("shortName") or info.get("longName"),
                    "GICS Sector": info.get("sector", None),
                    "GICS Sub-Industry": info.get("industry", None),
                })
            except Exception:
                recs.append({"Symbol": s, "Security": None, "GICS Sector": None, "GICS Sub-Industry": None})
        return pd.DataFrame.from_records(recs)


from typing import Optional, List


def select_sector_tickers(n: int = 20, prefer_sector: Optional[str] = None) -> List[str]:
    """Select N tickers from the same GICS Sector. Prefer a given sector if provided.

    If the preferred sector has fewer than N constituents, falls back to the largest sector.
    """
    try:
        cons = get_sp500_constituents()
        if prefer_sector is not None and (cons[cons["GICS Sector"] == prefer_sector].shape[0] >= n):
            sub = cons[cons["GICS Sector"] == prefer_sector]
        else:
            # Pick the sector with the most names
            top_sector = cons["GICS Sector"].value_counts().idxmax()
            sub = cons[cons["GICS Sector"] == top_sector]
        tickers = sorted(sub["Symbol"].dropna().astype(str).tolist())[:n]
        if len(tickers) >= n:
            return tickers
    except Exception:
        pass

    # Static fallback list (Information Technology, S&P 500 large caps)
    static_it = [
        "AAPL", "MSFT", "NVDA", "AVGO", "ADBE", "ORCL", "CSCO", "CRM", "INTC", "AMD",
        "TXN", "QCOM", "IBM", "AMAT", "MU", "LRCX", "NOW", "ADI", "INTU", "PANW",
    ]
    return static_it[:n]
