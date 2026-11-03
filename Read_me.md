# Knowledge-Augmented Financial Assistant Integrating Regulatory Reports and Live Market Data

An AI-powered financial research assistant that combines **RBI Annual Reports (2000–2026)** and **Yahoo Finance market data** into a single conversational interface.

Built using **Retrieval-Augmented Generation (RAG), LangGraph, Groq, FAISS, Hugging Face embeddings, and Streamlit**, the application retrieves regulatory information, analyzes financial market data, and generates evidence-based responses through an intelligent AI agent.

> ⚠️ **Disclaimer:** This project is for educational and research purposes only. It does not provide financial or investment advice.

---

## Overview

The Knowledge-Augmented Financial Assistant enables users to explore historical RBI monetary policy decisions, analyze stock market performance, compare companies, and retrieve financial information through natural language questions.

The application integrates two primary data sources:

* **RBI Annual Reports:** Retrieval-Augmented Generation over 27 annual reports, covering 2000–2026.
* **Yahoo Finance:** Historical market prices, company financials, dividends, analyst recommendations, and market news.

A LangGraph ReAct agent dynamically selects the appropriate tools based on user queries and generates responses using the information returned by those tools.

---

## Key Features

### Regulatory Research with RAG

* Retrieval over approximately 20,900 text chunks stored in a FAISS vector database.
* Year-aware RBI report retrieval.
* Contextual retrieval of monetary policy decisions and economic developments.
* Source passages displayed alongside generated answers.

### Historical Market Analysis

* Historical stock prices since 2000 where available.
* Index analysis including Nifty 50 and Sensex.
* Currency, gold, and crude oil historical data where supported.
* Year-by-year price performance and historical comparisons.

### Stock Comparison

* Compare 2–3 stocks or indices.
* Side-by-side financial and market metrics.
* Returns, valuation, profitability, and growth comparisons.

### Price Movement Analysis

* Analyze historical price increases and decreases.
* Compare stock performance against market indices.
* Identify significant daily price movements.
* Examine dividends, splits, annual profitability trends, recent news, and RBI context.
* Present possible explanations rather than claiming proven causes.

### Company Financial Analysis

* Company profiles and market capitalization.
* P/E, EPS, ROE, margins, debt, and beta where available.
* Annual and quarterly income statements.
* Balance sheets and cash flow statements.

### Additional Capabilities

* Dividend and stock split information.
* Analyst recommendations and price targets.
* Major and institutional holders.
* Recent financial news.
* Company ticker identification.
* Conversational follow-up questions within a session.

---

## Architecture

```text
                  User Question
                       |
                       v
              Streamlit Interface
                       |
                       v
            LangGraph ReAct Agent
                (Groq LLM)
                       |
          Tool Selection and Execution
                       |
       +---------------+----------------+
       |               |                |
       v               v                v
  RBI Report       Market Data      Financial
   Retrieval         Analysis         Analysis
       |               |                |
       v               v                v
  FAISS Vector     Yahoo Finance    Company Data
    Database         (yfinance)       & Metrics
       |               |                |
       +---------------+----------------+
                       |
                       v
              Retrieved Evidence
                       |
                       v
              AI-Generated Answer
                       |
                       v
             Streamlit Response
             + Source References
```

---

## Technology Stack

| Technology       | Purpose                                |
| ---------------- | -------------------------------------- |
| Python           | Core programming language              |
| Streamlit        | Interactive web application            |
| LangChain        | LLM integration and tool development   |
| LangGraph        | Agent orchestration and tool selection |
| Groq             | LLM inference                          |
| FAISS            | Vector similarity search               |
| Hugging Face     | Text embeddings                        |
| all-MiniLM-L6-v2 | Embedding model                        |
| yfinance         | Yahoo Finance market data              |
| python-dotenv    | Environment variable management        |

### Language Model

`openai/gpt-oss-120b` via Groq.

---

## Tools Implemented

| Tool                | Functionality                                   |
| ------------------- | ----------------------------------------------- |
| `RBIPolicyDatabase` | Search RBI reports with optional year filtering |
| `FindTicker`        | Identify Yahoo Finance symbols                  |
| `StockPriceHistory` | Historical prices and performance               |
| `CompareStocks`     | Compare multiple stocks or indices              |
| `PriceMoveAnalysis` | Analyze historical price movements              |
| `CompanyProfile`    | Retrieve company profile and valuation metrics  |
| `CompanyFinancials` | Retrieve financial statements                   |
| `StockEvents`       | Dividends, stock splits, earnings dates         |
| `AnalystAndHolders` | Analyst ratings and ownership information       |
| `StockNews`         | Retrieve recent news headlines                  |

---

## Example Questions

| Category            | Example                                                     |
| ------------------- | ----------------------------------------------------------- |
| RBI Monetary Policy | What did RBI do with the repo rate in 2015?                 |
| Economic Crisis     | How did RBI respond to the 2008–09 global financial crisis? |
| Historical Prices   | Analyze TCS share price performance from 2000 to 2026.      |
| Stock Comparison    | Compare TCS and Infosys.                                    |
| Market Analysis     | Why did Reliance stock rise in 2020?                        |
| Company Financials  | What are TCS net profit and market capitalization?          |
| Quarterly Results   | Show Infosys quarterly net profit.                          |
| Dividends           | What dividends did Reliance pay?                            |
| Analyst Research    | What are HDFC Bank analyst recommendations?                 |

---

## Project Structure

```text
Knowledge-Augmented-Financial-Assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
└── vector_db_rbi_final/
    ├── index.faiss
    └── index.pkl
```

The FAISS database is generated from the RBI Annual Reports and is required for RBI report retrieval.

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Trinadh555/Knowledge-Augmented-Financial-Assistant.git

cd Knowledge-Augmented-Financial-Assistant
```

### 2. Install Dependencies

Python 3.10 or newer is recommended.

```bash
pip install -r requirements.txt
```

### 3. Configure Groq API Key

Obtain an API key from:

[Groq Console](https://console.groq.com/)

Create a `.env` file in the project directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Alternatively, provide the API key through the application's sidebar or Streamlit secrets.

**Never commit your actual API key to GitHub.**

### 4. Configure RBI Vector Database

Ensure the `vector_db_rbi_final/` directory is available beside `app.py`.

The directory should contain:

```text
vector_db_rbi_final/
├── index.faiss
└── index.pkl
```

The vector database must be created using the same embedding model used by the application.

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## Building the RBI Vector Database

The RBI PDF documents are not included in this repository.

To rebuild the database:

1. Download RBI Annual Reports from the official RBI website.
2. Organize the PDFs by year.
3. Load documents using `PyPDFLoader`.
4. Split documents using `RecursiveCharacterTextSplitter`.
5. Generate embeddings using Hugging Face.
6. Create a FAISS index.
7. Save the database using `FAISS.save_local()`.

Example configuration:

```python
chunk_size = 1000
chunk_overlap = 200

embedding_model = "all-MiniLM-L6-v2"
```

Example database creation:

```python
vector_db = FAISS.from_documents(
    documents,
    embeddings
)

vector_db.save_local("vector_db_rbi_final")
```

Official RBI website:

https://www.rbi.org.in/

---

## Configuration

Important application settings include:

| Setting             | Description                          |
| ------------------- | ------------------------------------ |
| `MODEL_NAME`        | Groq model configuration             |
| `TOP_K`             | Number of RBI passages retrieved     |
| `CHUNK_CHARS`       | Context length per retrieved passage |
| `HISTORY_TURNS`     | Conversational memory                |
| `DATA_START`        | Historical market data start date    |
| `CACHE_TTL`         | Yahoo Finance cache duration         |
| `SHOW_SOURCES_BOX`  | Display retrieved RBI passages       |
| `SHOW_PAGE_NUMBERS` | Display PDF page references          |

Additional token budgets and session limits can be configured for demo usage.

---

## Deployment

The application can be deployed using:

* Streamlit Community Cloud
* Hugging Face Spaces

For public deployment:

* Configure API keys using platform secrets.
* Avoid exposing personal API credentials.
* Consider visitor rate limits and token budgets.
* Account for memory requirements of the embedding model and FAISS index.

---

## Limitations

* Groq free-tier usage is subject to rate limits.
* Yahoo Finance data may contain missing or unavailable fields.
* Historical company financial statements may be limited.
* RBI PDF tables may not be extracted perfectly.
* Historical market news may not be available for older periods.
* Price movements cannot be attributed to a single cause using market data alone.
* Market prices are based on available historical closes rather than real-time trading ticks.
* Important financial figures should be verified against official filings and RBI publications.

---

## Data Sources

* **Reserve Bank of India:** https://www.rbi.org.in/
* **Yahoo Finance:** https://finance.yahoo.com/
* **yfinance:** https://github.com/ranaroussi/yfinance

---

## Disclaimer

This project is developed for educational, research, and demonstration purposes.

The generated responses are not financial advice, investment recommendations, or guarantees of market performance.

Users should independently verify important information using official financial statements, RBI publications, and reliable market sources before making financial decisions.

---

## Author

**Mummana Trinadh**

* GitHub: https://github.com/Trinadh555
* LinkedIn: https://linkedin.com/in/mummana-trinadh

---

⭐ If you find this project interesting, consider starring the repository.
