import os
import time
from datetime import datetime, timedelta

import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

load_dotenv()  # local: reads GROQ_API_KEY from the .env file in this folder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "vector_db_rbi_final")  # folder with index.faiss + index.pkl
MODEL_NAME = "openai/gpt-oss-120b"  # or "openai/gpt-oss-20b" if you hit rate limits often

# ---- Display options ----
SHOW_PAGE_NUMBERS = False  # False = no PDF page numbers in answers or in the sources box
SHOW_SOURCES_BOX = True    # False = hide the "Sources used" expander completely

# ---- Retrieval / token settings (Groq free plan has a tokens-per-minute cap) ----
TOP_K = 5            # RBI chunks returned per search
CHUNK_CHARS = 800    # max characters kept from each chunk
HISTORY_TURNS = 4    # previous messages sent to the model
MAX_OUTPUT_TOKENS = 1200  # includes reasoning tokens for gpt-oss models

# ---- Yahoo Finance data settings ----
DATA_START = "2000-01-01"   # price history is downloaded from this date
CACHE_TTL = 6 * 3600        # keep downloaded Yahoo data for 6 hours (fewer slow calls)

# Safety limits, applied ONLY when the app uses the owner's key from Streamlit secrets
MAX_QUESTIONS_PER_SESSION = 15
COOLDOWN_SECONDS = 8
SESSION_TOKEN_LIMIT_DEMO = 30_000     # tokens per visitor session (owner's key only)
DAILY_TOKEN_BUDGET_DEMO = 150_000     # tokens per day for ALL visitors together (owner's key only)
                                      # keep this below your Groq daily limit (see console.groq.com > Limits)

# Per-minute guard: Groq's free plan showed a limit of 8000 tokens per minute for this model.
# Before each question the app waits until enough of the minute's budget is free.
# Set TPM_LIMIT = 0 to switch it off. Check console.groq.com > Limits for your real numbers.
TPM_LIMIT = 8000
EXPECTED_QUESTION_TOKENS = 4500   # rough cost of one question
MAX_AUTO_WAIT = 30                # longest automatic wait, in seconds

EXAMPLE_QUESTIONS = [
    "What did RBI do with the repo rate in 2015?",
    "Compare TCS and Infosys",
    "HDFC Bank vs ICICI Bank since 2010",
    "Why did the Nifty 50 fall in 2008?",
    "Why did Reliance stock rise in 2020?",
    "TCS share price performance from 2000 to 2026",
    "TCS net profit and market cap",
    "Latest news on Infosys",
]

CITE_RULE = (
    "- Cite the source file and page, e.g. (2015.pdf, p. 40)."
    if SHOW_PAGE_NUMBERS
    else "- Mention the report year (e.g. 'the 2015 report') but NEVER mention page numbers or 'p.' references."
)

SYSTEM_PROMPT = f"""You are Financial Chatbot, a professional financial research assistant
covering RBI annual reports (2000-2026) and Yahoo Finance market data.

Tools (each tool's description says what it returns):
- Use RBIPolicyDatabase for RBI, banking, inflation, repo rate, monetary policy and economy
  questions. If the user mentions a year, pass it as `year` (otherwise 0). For figures, include
  words like "per cent" or "averaged" in the query. The report numbered N covers FY (N-1) to N,
  so an event in 2016 usually appears in the 2017 report.
- Use FindTicker when unsure of a Yahoo symbol. Known symbols: Indian stocks use .NS (TCS.NS,
  RELIANCE.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS); Nifty 50 ^NSEI, Sensex ^BSESN, Bank Nifty
  ^NSEBANK, USD/INR USDINR=X, gold GC=F, crude oil CL=F, S&P 500 ^GSPC.
- StockPriceHistory: pass start_year (data starts in 2000) for year-by-year trends and returns.
- CompareStocks: any "A vs B", "compare", or "which is bigger / cheaper / better" question
  about 2 or 3 symbols.
- PriceMoveAnalysis: any "why did X rise / fall / increase / decrease / crash" question.
- If the user does not name a company, use the one from the earlier conversation; if there is
  none, ask which company they mean.

Accuracy rules:
- Answer ONLY from tool results. Never invent, estimate or calculate figures the tools did not
  return. If data is unavailable, say so and suggest the company's official filings.
- For "why did it rise / fall" questions: call PriceMoveAnalysis. When the period is in the past
  or macro conditions matter (rates, inflation, crisis), also call RBIPolicyDatabase for the
  relevant year. Answer with **What happened** (figures from the data) and **Possible factors**,
  listing only factors the data supports: market-wide move versus company-specific move (compare
  with the index), profit trend, dividends or splits, news headlines, RBI context. State clearly
  that these are possible factors, not proven causes. Never state a cause the tools did not return.
- Never say "the report does not contain X". Say "I could not find X in the retrieved passages",
  because you only see a few passages, not the whole report.
- Quote figures exactly, with units, currency and the date or period they refer to.
{CITE_RULE}

Answer format for RBI questions (clean Markdown):
**Summary:** one or two sentences with the direct answer.

**Key points**
- 3 to 5 concise bullets with exact figures and the year or period.

**Context:** one short sentence only if it adds real value; otherwise omit.

Answer format for market data:
**<Name> (<symbol>)**: headline figure first, then short bullets. Use the rupee symbol for
.NS / .BO symbols. Prices are last closes, not live ticks.
- Comparisons: a small Markdown table, then **Takeaway** with 2-3 factual bullets (who grew
  faster, who is cheaper on P/E, who is more profitable). Describe the data; never say which to buy.
- "Why" questions: **What happened**, then **Possible factors** as short bullets, then one line
  on what the data cannot confirm.
- Year-by-year data: a table.

Style rules:
- Formal, neutral, precise. No emojis, no filler, no "As an AI", no repeated disclaimers.
- If the user asks for buy/sell advice or predictions, politely explain that you provide
  information only, not investment advice, then share the relevant facts from the data.
- If a question is outside RBI reports and market data, say briefly what you can help with.
- Language: if the user writes Telugu in English letters (Tanglish), reply in clear, simple
  professional English. Use Telugu script only if the user writes in Telugu script. Otherwise
  reply in the language the user writes in."""

st.set_page_config(page_title="Knowledge-Augmented Financial Assistant Integrating Regulatory Reports and Live Market Data", page_icon="📈", layout="wide")
st.title("📈 Knowledge-Augmented Financial Assistant Integrating Regulatory Reports and Live Market Data")


def get_api_key():
    """Returns (key, source). Order: .env / environment -> Streamlit secrets -> sidebar."""
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key, "env"
    try:
        key = st.secrets["GROQ_API_KEY"]
        if key:
            own = st.sidebar.text_input(
                "Use your own Groq key (optional)",
                type="password",
                help="Removes the demo limits. Your key is used only in your session and is not stored.",
            )
            if own:
                return own, "user"
            return key, "secrets"
    except Exception:
        pass
    key = st.sidebar.text_input(
        "Enter your Groq API Key:",
        type="password",
        help="Free key from console.groq.com. It is used only in your session and is not stored.",
    )
    st.sidebar.markdown("[Get a free Groq API key](https://console.groq.com/keys)")
    return (key, "user") if key else (None, None)


api_key, key_source = get_api_key()
if not api_key:
    st.info(
        "Create a free key at [console.groq.com/keys](https://console.groq.com/keys) "
        "(no credit card needed) and paste it in the sidebar."
    )
    st.stop()

# Limits protect the owner's key on a public deployment
USE_LIMITS = key_source == "secrets"


@st.cache_resource
def load_knowledge_base():
    # Must be the SAME embedding model used when building the DB
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)


@st.cache_resource
def get_cache():
    """One dict that survives Streamlit reruns; holds downloaded Yahoo data."""
    return {}


vector_db = load_knowledge_base()
DATA_CACHE = get_cache()


# ---------------------------------------------------------------- helpers
def cached(key, fn, ttl=CACHE_TTL):
    """Tiny time-based cache so repeated questions do not re-download Yahoo data."""
    hit = DATA_CACHE.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    value = fn()
    DATA_CACHE[key] = (time.time(), value)
    return value


def is_indian(ticker: str) -> bool:
    return ticker.upper().endswith((".NS", ".BO"))


def money(value, ticker: str) -> str:
    """Large amounts: Indian tickers in rupees crore, others in billions."""
    if value is None or value != value:  # None or NaN
        return "n/a"
    if is_indian(ticker):
        return f"Rs {value / 1e7:,.0f} crore"
    return f"{value / 1e9:,.2f} billion"


def load_history(ticker: str):
    return cached(
        ("hist", ticker.upper()),
        lambda: yf.Ticker(ticker).history(start=DATA_START, auto_adjust=False),
    )


def load_info(ticker: str) -> dict:
    return cached(("info", ticker.upper()), lambda: yf.Ticker(ticker).info or {})


def numeric_score(text: str) -> int:
    """Chunks with more percentages are more likely to contain the actual figures."""
    return text.count("per cent") + text.count("%")


# ------------------------------------------------------------------ tools
@tool
def RBIPolicyDatabase(query: str, year: int = 0) -> str:
    """Search RBI reports about monetary policy, inflation, banking, repo rate, economy.
    Pass `year` (e.g. 2015) to restrict results to that year's report, or 0 for all years."""
    if year:
        # Over-fetch, keep that year's PDF, then prefer chunks that contain numbers
        docs = vector_db.similarity_search(query, k=60)
        docs = [d for d in docs if str(year) in d.metadata.get("source", "")][:12]
        docs = sorted(docs, key=lambda d: numeric_score(d.page_content), reverse=True)[:TOP_K]
    else:
        docs = vector_db.similarity_search(query, k=TOP_K)

    if not docs:
        return "No relevant RBI data found."

    parts = []
    for d in docs:
        src = d.metadata.get("source", "unknown").replace("\\", "/").split("/")[-1]
        if SHOW_PAGE_NUMBERS:
            page = d.metadata.get("page", -1) + 1
            header = f"[Source: {src}, page {page}]"
        else:
            header = f"[Source: {src}]"
        parts.append(f"{header}\n{d.page_content[:CHUNK_CHARS]}")
    return "\n\n---\n\n".join(parts)


@tool
def FindTicker(company_name: str) -> str:
    """Find Yahoo Finance symbols for a company name, e.g. 'Tata Motors'."""
    try:
        quotes = yf.Search(company_name, max_results=6).quotes
        if not quotes:
            return f"No symbols found for: {company_name}"
        lines = []
        for q in quotes:
            name = q.get("shortname") or q.get("longname") or ""
            lines.append(f"- {q.get('symbol')}: {name} ({q.get('exchDisp', '')}, {q.get('quoteType', '')})")
        return "\n".join(lines)
    except Exception as e:
        return f"Symbol search failed: {e}"


@tool
def StockPriceHistory(ticker: str, start_year: int = 0, end_year: int = 0) -> str:
    """Price data since Jan 2000 for any Yahoo symbol (stocks like TCS.NS, indices like ^NSEI,
    currencies like USDINR=X, commodities like GC=F). Returns latest close, all-time high and
    52-week range. If start_year is given (e.g. 2000), also returns year-by-year closes, yearly
    change, highs and lows from start_year to end_year (0 = latest year)."""
    try:
        hist = load_history(ticker)
        if hist.empty:
            return f"Could not find price history for symbol: {ticker}."

        last = hist["Close"].iloc[-1]
        last_date = hist.index[-1].strftime("%d %b %Y")
        ath = hist["High"].max()
        ath_date = hist["High"].idxmax().strftime("%d %b %Y")
        year52 = hist.tail(252)  # about one trading year
        hi52, hi52_date = year52["High"].max(), year52["High"].idxmax().strftime("%d %b %Y")
        lo52, lo52_date = year52["Low"].min(), year52["Low"].idxmin().strftime("%d %b %Y")

        lines = [
            f"{ticker}, data available from {hist.index[0].strftime('%b %Y')}:",
            f"- Latest close: {last:.2f} on {last_date}",
            f"- All-time high (intraday): {ath:.2f} on {ath_date} "
            f"(latest close is {(last / ath - 1) * 100:+.1f}% vs it)",
            f"- 52-week high: {hi52:.2f} on {hi52_date}; 52-week low: {lo52:.2f} on {lo52_date}",
        ]

        if start_year:
            end_y = end_year or hist.index[-1].year
            start_y = max(start_year, hist.index[0].year)
            sub = hist[(hist.index.year >= start_y) & (hist.index.year <= end_y)]
            if sub.empty:
                lines.append(f"- No data between {start_year} and {end_y}.")
            else:
                year_close_all = hist.groupby(hist.index.year)["Close"].last()
                grp = sub.groupby(sub.index.year)
                closes, highs, lows = grp["Close"].last(), grp["High"].max(), grp["Low"].min()
                lines.append(f"Year-by-year ({start_y}-{end_y}), change vs previous year-end close:")
                for y, c in closes.items():
                    prev = year_close_all.get(y - 1)
                    chg = f"{(c / prev - 1) * 100:+.0f}%" if prev else "n/a"
                    lines.append(f"  {y}: close {c:.2f} ({chg}), high {highs[y]:.2f}, low {lows[y]:.2f}")
                first, final = sub["Close"].iloc[0], sub["Close"].iloc[-1]
                years = (sub.index[-1] - sub.index[0]).days / 365.25
                summary = f"- Change over the period: {(final / first - 1) * 100:+.1f}%"
                if years >= 1:
                    summary += f" (about {((final / first) ** (1 / years) - 1) * 100:.1f}% per year, price only)"
                lines.append(summary)

        lines.append("Prices are adjusted for splits and bonuses by Yahoo Finance; dividends not included.")
        return "\n".join(lines)
    except Exception as e:
        return f"Error fetching price history: {e}"


@tool
def CompanyProfile(ticker: str) -> str:
    """Company profile and key ratios: sector, industry, market cap, P/E, EPS, price-to-book,
    ROE, margins, debt-to-equity, beta, employees and a short description."""
    try:
        info = load_info(ticker)
        if not info:
            return f"Could not find company data for symbol: {ticker}."
        lines = [f"{info.get('longName') or ticker} ({ticker})"]

        def add(label, value):
            if value not in (None, ""):
                lines.append(f"- {label}: {value}")

        add("Sector / industry", f"{info.get('sector', 'n/a')} / {info.get('industry', 'n/a')}")
        add("Country", info.get("country"))
        add("Employees", f"{info['fullTimeEmployees']:,}" if info.get("fullTimeEmployees") else None)
        add("Market cap", money(info["marketCap"], ticker) if info.get("marketCap") else None)
        add("Trailing P/E", f"{info['trailingPE']:.1f}" if info.get("trailingPE") else None)
        add("Forward P/E", f"{info['forwardPE']:.1f}" if info.get("forwardPE") else None)
        add("EPS (trailing)", f"{info['trailingEps']:.2f}" if info.get("trailingEps") else None)
        add("Price to book", f"{info['priceToBook']:.2f}" if info.get("priceToBook") else None)
        add("Dividend per share (annual)", f"{info['dividendRate']:.2f}" if info.get("dividendRate") else None)
        add("Return on equity", f"{info['returnOnEquity'] * 100:.1f}%" if info.get("returnOnEquity") else None)
        add("Profit margin", f"{info['profitMargins'] * 100:.1f}%" if info.get("profitMargins") else None)
        add("Debt to equity", f"{info['debtToEquity']:.1f}" if info.get("debtToEquity") else None)
        add("Beta", f"{info['beta']:.2f}" if info.get("beta") else None)
        add("Website", info.get("website"))
        if info.get("longBusinessSummary"):
            lines.append(f"- About: {info['longBusinessSummary'][:300]}...")
        lines.append("Source: Yahoo Finance (unaudited aggregation; may lag official filings).")
        return "\n".join(lines)
    except Exception as e:
        return f"Error fetching company profile: {e}"


STATEMENT_ROWS = {
    "income": [
        ("Total Revenue", "Revenue", "money"), ("Gross Profit", "Gross profit", "money"),
        ("Operating Income", "Operating income", "money"), ("EBITDA", "EBITDA", "money"),
        ("Net Income", "Net profit", "money"), ("Diluted EPS", "Diluted EPS", "plain"),
    ],
    "balance": [
        ("Total Assets", "Total assets", "money"),
        ("Total Liabilities Net Minority Interest", "Total liabilities", "money"),
        ("Stockholders Equity", "Shareholders' equity", "money"), ("Total Debt", "Total debt", "money"),
        ("Cash And Cash Equivalents", "Cash and equivalents", "money"),
    ],
    "cashflow": [
        ("Operating Cash Flow", "Operating cash flow", "money"),
        ("Investing Cash Flow", "Investing cash flow", "money"),
        ("Financing Cash Flow", "Financing cash flow", "money"),
        ("Free Cash Flow", "Free cash flow", "money"), ("Capital Expenditure", "Capital expenditure", "money"),
    ],
}


@tool
def CompanyFinancials(ticker: str, statement: str = "income", quarterly: bool = False) -> str:
    """Financial statements for a company. statement is 'income' (revenue, net profit, EPS),
    'balance' (assets, debt, equity) or 'cashflow'. Set quarterly=True for quarterly figures."""
    try:
        t = yf.Ticker(ticker)
        kind = statement.lower().replace(" ", "").replace("_", "")
        if kind.startswith("bal"):
            kind, df = "balance", cached(("bs", ticker.upper(), quarterly), lambda: t.quarterly_balance_sheet if quarterly else t.balance_sheet)
        elif kind.startswith("cash"):
            kind, df = "cashflow", cached(("cf", ticker.upper(), quarterly), lambda: t.quarterly_cashflow if quarterly else t.cashflow)
        else:
            kind, df = "income", cached(("is", ticker.upper(), quarterly), lambda: t.quarterly_income_stmt if quarterly else t.income_stmt)

        if df is None or df.empty:
            return f"{kind} statement is not available from Yahoo for {ticker}."

        lines = [f"{ticker} {kind} statement ({'quarterly' if quarterly else 'annual'}), periods ending:"]
        for label, title, fmt in STATEMENT_ROWS[kind]:
            if label in df.index:
                row = df.loc[label].dropna().head(4)
                if len(row):
                    vals = []
                    for d, v in row.items():
                        shown = money(v, ticker) if fmt == "money" else f"{v:.2f}"
                        vals.append(f"{d.strftime('%b %Y')}: {shown}")
                    lines.append(f"- {title}: " + "; ".join(vals))
        if len(lines) == 1:
            return f"No standard line items found for {ticker}."
        lines.append("Source: Yahoo Finance (unaudited aggregation; may lag official filings).")
        return "\n".join(lines)
    except Exception as e:
        return f"Error fetching financial statements: {e}"


@tool
def StockEvents(ticker: str) -> str:
    """Dividend history (recent payments and yearly totals), stock splits and the next
    earnings date for a company."""
    t = yf.Ticker(ticker)
    lines = [f"{ticker} corporate events:"]
    try:
        div = t.dividends
        if div is not None and len(div):
            recent = "; ".join(f"{d.strftime('%d %b %Y')}: {v:.2f}" for d, v in div.tail(6).items())
            yearly = div.groupby(div.index.year).sum().tail(10)
            totals = "; ".join(f"{y}: {v:.2f}" for y, v in yearly.items())
            lines.append(f"- Recent dividends per share: {recent}")
            lines.append(f"- Dividend per share by calendar year: {totals}")
        else:
            lines.append("- No dividend history found.")
    except Exception as e:
        lines.append(f"- Dividends unavailable ({e})")
    try:
        splits = t.splits
        if splits is not None and len(splits):
            lines.append("- Splits/bonus ratios: " + "; ".join(f"{d.strftime('%d %b %Y')}: {v:g}-for-1" for d, v in splits.items()))
        else:
            lines.append("- No stock splits found.")
    except Exception as e:
        lines.append(f"- Splits unavailable ({e})")
    try:
        cal = t.calendar
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if dates:
            lines.append("- Next earnings date: " + ", ".join(str(d) for d in dates))
    except Exception:
        pass
    return "\n".join(lines)


@tool
def AnalystAndHolders(ticker: str) -> str:
    """Analyst recommendations, analyst price targets, major holders and top institutional
    holders for a company."""
    t = yf.Ticker(ticker)
    lines = [f"{ticker} analyst and ownership data:"]
    try:
        rec = t.recommendations_summary
        if rec is not None and not rec.empty:
            r = rec.iloc[0]
            lines.append(
                f"- Analyst ratings now: strong buy {r.get('strongBuy', 0)}, buy {r.get('buy', 0)}, "
                f"hold {r.get('hold', 0)}, sell {r.get('sell', 0)}, strong sell {r.get('strongSell', 0)}"
            )
    except Exception:
        pass
    try:
        pt = t.analyst_price_targets
        if pt:
            lines.append(
                f"- Price targets: low {pt.get('low')}, mean {pt.get('mean')}, "
                f"median {pt.get('median')}, high {pt.get('high')} (current {pt.get('current')})"
            )
    except Exception:
        pass
    try:
        mh = t.major_holders
        if mh is not None and not mh.empty:
            lines.append("- Major holders: " + "; ".join(f"{i}: {float(v[0]) * 100:.1f}%" for i, v in mh.iterrows()))
    except Exception:
        pass
    try:
        ih = t.institutional_holders
        if ih is not None and not ih.empty:
            top = ih.head(5)
            lines.append("- Top institutional holders: " + "; ".join(
                f"{row['Holder']} ({row['pctHeld'] * 100:.1f}%)" for _, row in top.iterrows()))
    except Exception:
        pass
    if len(lines) == 1:
        return f"No analyst or holder data available from Yahoo for {ticker}."
    return "\n".join(lines)


@tool
def StockNews(ticker: str) -> str:
    """Recent news headlines for a company or index symbol from Yahoo Finance."""
    try:
        items = yf.Ticker(ticker).news or []
        if not items:
            return f"No recent news found for {ticker}."
        lines = [f"Recent news for {ticker}:"]
        for it in items[:5]:
            c = it.get("content", it)
            title = c.get("title", "")
            provider = (c.get("provider") or {}).get("displayName") or it.get("publisher", "")
            date = str(c.get("pubDate", ""))[:10]
            lines.append(f"- {title} ({provider}, {date})")
        return "\n".join(lines)
    except Exception as e:
        return f"Error fetching news: {e}"


def pct_change_over(hist, years: int):
    """% price change over the last N years, or None if there is not enough history."""
    end = hist.index[-1]
    target = end - timedelta(days=int(365.25 * years))
    if hist.index[0] > target + timedelta(days=15):
        return None
    base = hist[hist.index >= target]["Close"].iloc[0]
    return (hist["Close"].iloc[-1] / base - 1) * 100


def annual_income(ticker: str):
    return cached(("is", ticker.upper(), False), lambda: yf.Ticker(ticker).income_stmt)


def is_stock_symbol(sym: str) -> bool:
    return not (sym.startswith("^") or "=" in sym)


COMPARE_ROWS = [
    "Latest close", "1-year price change", "3-year price change", "5-year price change",
    "10-year price change", "Since listing / 2000", "All-time high", "Below all-time high",
    "52-week range", "Market cap", "Trailing P/E", "Return on equity", "Profit margin",
    "Latest annual revenue", "Latest annual net profit", "Net profit growth (latest FY)",
]


@tool
def CompareStocks(tickers: str) -> str:
    """Side-by-side comparison of 2 or 3 stocks or indices (comma separated Yahoo symbols, e.g.
    'TCS.NS,INFY.NS'): price change over 1/3/5/10 years and since 2000, all-time high, 52-week
    range, market cap, P/E, ROE, profit margin, revenue and net profit. Returns a table."""
    try:
        symbols = [s.strip().upper() for s in tickers.split(",") if s.strip()][:3]
        if len(symbols) < 2:
            return "Please give at least two symbols separated by commas."

        data = {}
        for sym in symbols:
            hist = load_history(sym)
            if hist.empty:
                data[sym] = {}
                continue
            d = {}
            last = hist["Close"].iloc[-1]
            d["Latest close"] = f"{last:.2f} ({hist.index[-1].strftime('%d %b %Y')})"
            for y in (1, 3, 5, 10):
                r = pct_change_over(hist, y)
                d[f"{y}-year price change"] = f"{r:+.1f}%" if r is not None else "n/a"
            first = hist["Close"].iloc[0]
            yrs = (hist.index[-1] - hist.index[0]).days / 365.25
            if yrs >= 1:
                d["Since listing / 2000"] = (
                    f"{(last / first - 1) * 100:+.0f}% (~{((last / first) ** (1 / yrs) - 1) * 100:.1f}%/yr) "
                    f"from {hist.index[0].strftime('%b %Y')}"
                )
            ath = hist["High"].max()
            d["All-time high"] = f"{ath:.2f} ({hist['High'].idxmax().strftime('%b %Y')})"
            d["Below all-time high"] = f"{(last / ath - 1) * 100:.1f}%"
            y52 = hist.tail(252)
            d["52-week range"] = f"{y52['Low'].min():.2f} - {y52['High'].max():.2f}"

            if is_stock_symbol(sym):
                try:
                    info = load_info(sym)
                    if info.get("marketCap"):
                        d["Market cap"] = money(info["marketCap"], sym)
                    if info.get("trailingPE"):
                        d["Trailing P/E"] = f"{info['trailingPE']:.1f}"
                    if info.get("returnOnEquity"):
                        d["Return on equity"] = f"{info['returnOnEquity'] * 100:.1f}%"
                    if info.get("profitMargins"):
                        d["Profit margin"] = f"{info['profitMargins'] * 100:.1f}%"
                except Exception:
                    pass
                try:
                    inc = annual_income(sym)
                    if inc is not None and not inc.empty:
                        if "Total Revenue" in inc.index:
                            rev = inc.loc["Total Revenue"].dropna()
                            if len(rev):
                                d["Latest annual revenue"] = f"{money(rev.iloc[0], sym)} (FY {rev.index[0].strftime('%b %Y')})"
                        if "Net Income" in inc.index:
                            ni = inc.loc["Net Income"].dropna()
                            if len(ni):
                                d["Latest annual net profit"] = f"{money(ni.iloc[0], sym)} (FY {ni.index[0].strftime('%b %Y')})"
                            if len(ni) >= 2 and ni.iloc[1]:
                                d["Net profit growth (latest FY)"] = f"{(ni.iloc[0] / ni.iloc[1] - 1) * 100:+.1f}%"
                except Exception:
                    pass
            data[sym] = d

        lines = ["| Metric | " + " | ".join(symbols) + " |", "|---|" + "---|" * len(symbols)]
        for row in COMPARE_ROWS:
            vals = [data[s].get(row, "n/a") for s in symbols]
            if any(v != "n/a" for v in vals):
                lines.append(f"| {row} | " + " | ".join(vals) + " |")
        lines.append("Prices are split/bonus adjusted by Yahoo Finance, dividends not included. "
                     "Financials are unaudited aggregations that may lag official filings.")
        return "\n".join(lines)
    except Exception as e:
        return f"Error comparing stocks: {e}"


@tool
def PriceMoveAnalysis(ticker: str, start_date: str = "", end_date: str = "") -> str:
    """Evidence for why a stock or index rose or fell between two dates (YYYY-MM-DD; default is the
    last 12 months): total change, comparison with the market index, peak and trough, biggest
    single-day moves, dividends/splits in the window, annual net profit trend and recent news.
    Use for 'why did X increase / decrease / fall / rise' questions."""
    try:
        hist = load_history(ticker)
        if hist.empty:
            return f"Could not find price history for symbol: {ticker}."

        try:
            ed = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else hist.index[-1].date()
            sd = datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else ed - timedelta(days=365)
        except ValueError:
            return "Dates must be in YYYY-MM-DD format."

        def window(h):
            return h[(h.index.date >= sd) & (h.index.date <= ed)]

        win = window(hist)
        if len(win) < 2:
            return (f"No price data for {ticker} between {sd} and {ed}. "
                    f"Data available from {hist.index[0].strftime('%d %b %Y')}.")

        first_d, last_d = win.index[0], win.index[-1]
        start_close, end_close = win["Close"].iloc[0], win["Close"].iloc[-1]
        chg = (end_close / start_close - 1) * 100
        lines = [
            f"{ticker} from {first_d.strftime('%d %b %Y')} to {last_d.strftime('%d %b %Y')} "
            f"({len(win)} trading days):",
            f"- Price change: {start_close:.2f} -> {end_close:.2f} = {chg:+.1f}%",
        ]

        # Compare with the market to separate market-wide moves from company-specific ones
        up = ticker.upper()
        bench = "^NSEI" if is_indian(up) else ("^GSPC" if is_stock_symbol(up) else "")
        if bench and up != bench:
            try:
                bw = window(load_history(bench))
                if len(bw) >= 2:
                    bchg = (bw["Close"].iloc[-1] / bw["Close"].iloc[0] - 1) * 100
                    lines.append(
                        f"- Market index {bench} over the same dates: {bchg:+.1f}% "
                        f"(stock vs index: {chg - bchg:+.1f} percentage points)"
                    )
            except Exception:
                pass

        lines.append(f"- Peak (intraday high): {win['High'].max():.2f} on {win['High'].idxmax().strftime('%d %b %Y')}; "
                     f"trough (intraday low): {win['Low'].min():.2f} on {win['Low'].idxmin().strftime('%d %b %Y')}")
        dd = win["Close"] / win["Close"].cummax() - 1
        lines.append(f"- Largest fall from a running peak: {dd.min() * 100:.1f}% (bottom on {dd.idxmin().strftime('%d %b %Y')})")

        rets = win["Close"].pct_change().dropna() * 100
        lines.append("- Biggest single-day gains: " + "; ".join(f"{d.strftime('%d %b %Y')} {v:+.1f}%" for d, v in rets.nlargest(3).items()))
        lines.append("- Biggest single-day falls: " + "; ".join(f"{d.strftime('%d %b %Y')} {v:+.1f}%" for d, v in rets.nsmallest(3).items()))

        if is_stock_symbol(up):
            t = yf.Ticker(ticker)
            try:
                div = t.dividends
                dw = div[(div.index.date >= sd) & (div.index.date <= ed)] if div is not None and len(div) else []
                if len(dw):
                    lines.append(f"- Dividends in the window: {len(dw)} payment(s), total {dw.sum():.2f} per share")
                spl = t.splits
                sw = spl[(spl.index.date >= sd) & (spl.index.date <= ed)] if spl is not None and len(spl) else []
                if len(sw):
                    lines.append("- Splits/bonus in the window: " + "; ".join(f"{d.strftime('%d %b %Y')} {v:g}-for-1" for d, v in sw.items()))
            except Exception:
                pass
            try:
                inc = annual_income(ticker)
                if inc is not None and "Net Income" in inc.index:
                    ni = inc.loc["Net Income"].dropna()
                    near = [(d, v) for d, v in ni.items()
                            if sd - timedelta(days=400) <= d.date() <= ed + timedelta(days=400)]
                    if near:
                        lines.append("- Annual net profit around this period: " + "; ".join(
                            f"FY {d.strftime('%b %Y')}: {money(v, ticker)}" for d, v in sorted(near)))
                    else:
                        lines.append("- Annual net profit for this period is not available from Yahoo (it keeps only recent years).")
            except Exception:
                pass
            if (datetime.now().date() - ed).days <= 120:
                try:
                    for it in (t.news or [])[:5]:
                        c = it.get("content", it)
                        lines.append(f"- News: {c.get('title', '')} ({str(c.get('pubDate', ''))[:10]})")
                except Exception:
                    pass
            else:
                lines.append("- News headlines for older periods are not available from Yahoo.")

        lines.append("These are data points, not proven causes: price data alone cannot prove why a price moved.")
        return "\n".join(lines)
    except Exception as e:
        return f"Error analysing price move: {e}"


ALL_TOOLS = [
    RBIPolicyDatabase, FindTicker, StockPriceHistory, CompareStocks, PriceMoveAnalysis,
    CompanyProfile, CompanyFinancials, StockEvents, AnalystAndHolders, StockNews,
]


@st.cache_resource
def build_agent(key: str):
    llm = ChatGroq(
        model=MODEL_NAME,
        api_key=key,
        temperature=0,
        max_tokens=MAX_OUTPUT_TOKENS,
        max_retries=3,  # auto-retries short rate-limit waits
    )
    return create_react_agent(llm, ALL_TOOLS, prompt=SYSTEM_PROMPT)


agent_executor = build_agent(api_key)


def to_text(content) -> str:
    """Some wrappers return a list of content blocks instead of a string."""
    if isinstance(content, str):
        return content
    return "".join(
        p.get("text", "") if isinstance(p, dict) else str(p) for p in content
    )


def _tpm_events(key: str) -> list:
    """(timestamp, tokens) records for one API key, shared across sessions."""
    return DATA_CACHE.setdefault(f"tpm_{hash(key)}", [])


def tokens_last_minute(key: str) -> int:
    now = time.time()
    events = _tpm_events(key)
    events[:] = [(t, n) for t, n in events if now - t < 60]
    return sum(n for _, n in events)


def record_tokens(key: str, tokens: int):
    _tpm_events(key).append((time.time(), tokens))


def seconds_until_room(key: str, needed: int) -> float:
    """Seconds to wait until `needed` tokens fit under the per-minute limit (0 = go now)."""
    needed = min(needed, TPM_LIMIT)
    over = tokens_last_minute(key) + needed - TPM_LIMIT
    if over <= 0:
        return 0.0
    freed = 0
    for t, n in sorted(_tpm_events(key)):
        freed += n
        if freed >= over:
            return max(0.0, t + 60 - time.time())
    return 60.0


def daily_usage() -> dict:
    """Tokens used today by everyone (shared across sessions, resets each day)."""
    today = datetime.now().strftime("%Y-%m-%d")
    usage = DATA_CACHE.get("usage")
    if not usage or usage["date"] != today:
        usage = {"date": today, "tokens": 0}
        DATA_CACHE["usage"] = usage
    return usage


def tokens_in(response) -> int:
    """Total tokens (input + output) used by all model calls in this answer."""
    total = 0
    for msg in response["messages"]:
        meta = getattr(msg, "usage_metadata", None)
        if meta:
            total += meta.get("total_tokens", 0)
    if total == 0:  # fallback estimate if the provider sent no usage info
        total = sum(len(to_text(m.content)) for m in response["messages"]) // 4
    return total


def render_usage(placeholder, limit: int):
    """Draw the token meter in the sidebar (called again after each answer)."""
    used = st.session_state.tokens_used
    with placeholder.container():
        if limit:
            st.progress(min(used / limit, 1.0))
            st.caption(f"{used:,} / {limit:,} tokens used this session")
        else:
            st.caption(f"{used:,} tokens used this session (no limit)")
        if TPM_LIMIT:
            st.caption(f"Last 60 seconds: {tokens_last_minute(api_key):,} / {TPM_LIMIT:,} tokens")
        if USE_LIMITS:
            share = min(daily_usage()["tokens"] / DAILY_TOKEN_BUDGET_DEMO, 1.0)
            st.caption(f"Demo capacity used today: {share * 100:.0f}%")


def show_sources(sources):
    """Show the exact RBI passages the model saw for this answer."""
    if SHOW_SOURCES_BOX and sources:
        with st.expander("📄 Sources used (retrieved RBI passages)"):
            for s in sources:
                st.text(s)


# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "question_count" not in st.session_state:
    st.session_state.question_count = 0
if "last_question_time" not in st.session_state:
    st.session_state.last_question_time = 0.0
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "tokens_used" not in st.session_state:
    st.session_state.tokens_used = 0

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("Usage")
    if USE_LIMITS:
        session_limit = SESSION_TOKEN_LIMIT_DEMO
    else:
        session_limit = st.slider(
            "Session token limit (0 = no limit)",
            0, 200_000, 0, step=5_000,
            help="Stops new questions once this session has used this many tokens. "
                 "Protects your free Groq quota.",
        )
    usage_ph = st.empty()
    st.divider()
    st.header("About")
    st.write(
        "Answers RBI policy and economy questions from 27 annual reports, and uses "
        "Yahoo Finance for prices since 2000, financials, dividends, analyst views and news."
    )
    st.caption("For educational purposes only. Not financial advice.")
    st.subheader("Try asking")
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        if st.button(q, key=f"example_{i}", use_container_width=True):
            st.session_state.pending_query = q
    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption("Tip: the RBI report numbered N covers FY (N-1) to N. Mention the year for better results.")

render_usage(usage_ph, session_limit)

# ---------- Chat history ----------
if not st.session_state.messages:
    st.info(
        "**Welcome.** Ask about RBI policy, inflation, repo rate, banking or the economy "
        "for a specific year, or ask about stocks, indices and companies: prices since 2000, "
        "net profit, dividends, analyst views and news. Use the examples in the sidebar to start."
    )

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        show_sources(m.get("sources"))

typed_query = st.chat_input("Ask about RBI reports, stocks or companies...")
user_query = typed_query or st.session_state.pending_query
st.session_state.pending_query = None

if user_query:
    blocked = None
    if USE_LIMITS:
        wait = COOLDOWN_SECONDS - (time.time() - st.session_state.last_question_time)
        if st.session_state.question_count >= MAX_QUESTIONS_PER_SESSION:
            blocked = (
                f"Demo limit reached ({MAX_QUESTIONS_PER_SESSION} questions). "
                "Enter your own Groq key in the sidebar, or refresh the page."
            )
        elif wait > 0:
            blocked = f"Please wait {int(wait) + 1} seconds before the next question."

    if not blocked and USE_LIMITS and daily_usage()["tokens"] >= DAILY_TOKEN_BUDGET_DEMO:
        blocked = (
            "Today's demo capacity is used up. Enter your own Groq key in the sidebar, "
            "or try again tomorrow."
        )
    if not blocked and session_limit and st.session_state.tokens_used >= session_limit:
        blocked = (
            f"Session token limit reached ({session_limit:,} tokens). "
            + ("Enter your own Groq key in the sidebar to continue."
               if USE_LIMITS else "Raise the limit in the sidebar to continue.")
        )

    if blocked:
        st.warning(blocked)
    else:
        st.session_state.question_count += 1
        st.session_state.last_question_time = time.time()

        st.chat_message("user").markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("Researching..."):
                try:
                    history = []
                    for m in st.session_state.messages[-HISTORY_TURNS:]:
                        cls = HumanMessage if m["role"] == "user" else AIMessage
                        history.append(cls(content=m["content"]))

                    if TPM_LIMIT:
                        wait = seconds_until_room(api_key, EXPECTED_QUESTION_TOKENS)
                        if wait > 0:
                            with st.spinner(f"Waiting {int(wait) + 1}s for the per-minute token limit..."):
                                time.sleep(min(wait + 1, MAX_AUTO_WAIT))

                    response = agent_executor.invoke({"messages": history})

                    used_now = tokens_in(response)
                    record_tokens(api_key, used_now)
                    st.session_state.tokens_used += used_now
                    daily_usage()["tokens"] += used_now
                    render_usage(usage_ph, session_limit)
                    output_text = to_text(response["messages"][-1].content)

                    # History has no tool messages, so these are all from this turn
                    sources = [
                        to_text(msg.content)
                        for msg in response["messages"]
                        if isinstance(msg, ToolMessage) and msg.name == "RBIPolicyDatabase"
                    ]

                    st.markdown(output_text)
                    show_sources(sources)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": output_text, "sources": sources}
                    )
                except Exception as e:
                    if "429" in str(e) or "rate_limit" in str(e).lower():
                        st.warning(
                            "Rate limit reached (Groq free plan). "
                            "Please wait 15-20 seconds and try again."
                        )
                    else:
                        st.error(f"Error: {e}")