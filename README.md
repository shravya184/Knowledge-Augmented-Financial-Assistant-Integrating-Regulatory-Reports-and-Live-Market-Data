# Knowledge-Augmented Financial Assistant

An AI-powered financial research assistant that combines **RBI Annual Reports (2000–2026)** and **Yahoo Finance market data** into a single conversational interface.

Built using **Retrieval-Augmented Generation (RAG), LangGraph, Groq, FAISS, Hugging Face embeddings, and Streamlit**, this application retrieves regulatory information, analyzes financial market data, and generates evidence-based responses through an intelligent AI agent.

> **Disclaimer:** This project is for educational and research purposes only. It does not provide financial or investment advice.

## Overview

The Knowledge-Augmented Financial Assistant helps users explore historical RBI monetary policy decisions, analyze stock market performance, compare companies, and retrieve financial information through natural-language questions.

It integrates two primary data sources:

- **RBI Annual Reports:** RAG-based retrieval over annual reports covering 2000–2026.
- **Yahoo Finance:** Historical market prices, company financials, dividends, analyst recommendations, and market news.

A LangGraph ReAct agent dynamically selects appropriate tools based on user queries and generates responses using the information returned by those tools.

## Key Features

### 1. Regulatory Research with RAG
- Retrieval over approximately 20,900 text chunks stored in a FAISS vector database.
- Year-aware RBI report retrieval.
- Contextual retrieval of monetary policy decisions and economic developments.
- Source passages displayed alongside generated answers.

### 2. Historical Market Analysis
- Historical stock prices since 2000, where available.
- Nifty 50 and Sensex analysis.
- Currency, gold, and crude oil historical data, where supported.
- Year-by-year price performance and historical comparisons.

### 3. Stock Comparison
- Compare 2–3 stocks or indices.
- Side-by-side financial and market metrics.
- Returns, valuation, profitability, and growth comparisons.

### 4. Price Movement Analysis
- Analyze historical price increases and decreases.
- Compare stock performance against market indices.
- Identify significant daily price movements.
- Examine dividends, stock splits, annual profitability trends, recent news, and RBI context.
- Present possible explanations without claiming that correlation proves causation.

### 5. Company Financial Analysis
- Company profiles and market capitalization.
- P/E, EPS, ROE, margins, debt, and beta, where available.
- Annual and quarterly income statements.
- Balance sheets and cash flow statements.

### 6. Additional Capabilities
- Dividend and stock split information.
- Analyst recommendations and price targets.
- Major and institutional holders.
- Recent financial news.
- Company ticker identification.
- Conversational follow-up questions within a session.

## Architecture

```text
User Question
     |
     v
Streamlit Interface
     |
     v
LangGraph ReAct Agent (Groq LLM)
     |
     v
Tool Selection and Execution
     |
     +------------------+------------------+
     |                  |                  |
     v                  v                  v
RBI Report         Market Data        Financial
Retrieval          Analysis           Analysis
     |                  |                  |
     v                  v                  v
FAISS Vector       Yahoo Finance      Company Data
Database           (yfinance)         and Metrics
     |                  |                  |
     +------------------+------------------+
                        |
                        v
                Retrieved Evidence
                        |
                        v
                 AI-Generated Answer
                        |
                        v
                Response with Sources
```

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| LangChain | LLM integration and tool development |
| LangGraph | Agent orchestration and tool selection |
| Groq | LLM inference |
| FAISS | Vector similarity search |
| Hugging Face | Text embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| yfinance | Yahoo Finance market data |
| python-dotenv | Environment variable management |

### Language Model

`openai/gpt-oss-120b` via Groq.

## Tools Implemented

| Tool | Functionality |
|---|---|
| `RBIPolicyDatabase` | Search RBI reports with optional year filtering |
| `FindTicker` | Identify Yahoo Finance symbols |
| `StockPriceHistory` | Historical prices and performance |
| `CompareStocks` | Compare multiple stocks or indices |
| `PriceMoveAnalysis` | Analyze historical price movements |
| `CompanyProfile` | Retrieve company profiles and valuation metrics |
| `CompanyFinancials` | Retrieve financial statements |
| `StockEvents` | Dividends, stock splits, and earnings dates |
| `AnalystAndHolders` | Analyst ratings and ownership information |
| `StockNews` | Retrieve recent news headlines |

## Example Questions

Try asking the assistant questions such as:

- What did RBI do with the repo rate in 2015?
- How did RBI respond to the 2008–09 global financial crisis?
- Analyze TCS share price performance from 2000 to 2026.
- Compare TCS and Infosys.
- Why did Reliance stock rise in 2020?
- What are TCS's net profit and market capitalization?
- Show Infosys quarterly net profit.
- What dividends did Reliance pay?
- What are HDFC Bank analyst recommendations?

## Project Structure

```text
Knowledge-Augmented-Financial-Assistant/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
└── vector_db_rbi_final/
    ├── index.faiss
    └── index.pkl
```

The FAISS database is used for RBI report retrieval. Ensure the index files are available if you want to use the prebuilt database.

## Installation and Setup

### Prerequisites

- Python 3.10 or newer.
- Git.
- A Groq API key.
- The project source code and required vector database files.

### 1. Clone the Repository

```bash
git clone https://github.com/shravya184/Knowledge-Augmented-Financial-Assistant-Integrating-Regulatory-Reports-and-Live-Market-Data.git

cd Knowledge-Augmented-Financial-Assistant-Integrating-Regulatory-Reports-and-Live-Market-Data
```

If you are using your own GitHub repository, replace the repository URL with your repository's URL.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the Groq API Key

Obtain an API key from the [Groq Console](https://console.groq.com/).

Create a `.env` file in the project directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Alternatively, provide the API key through the application's sidebar or Streamlit secrets, if supported by your implementation.

**Never commit your actual API key to GitHub.**

### 4. Configure the RBI Vector Database

Ensure the `vector_db_rbi_final/` directory is available beside `app.py`.

```text
vector_db_rbi_final/
├── index.faiss
└── index.pkl
```

The vector database must be compatible with the embedding model used by the application. Only load pickle files from trusted sources.

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## Building the RBI Vector Database

The original RBI PDF documents are not included in this repository.

To rebuild the database:

1. Download RBI Annual Reports from the [official RBI website](https://www.rbi.org.in/).
2. Organize the PDF files by year.
3. Load the documents using `PyPDFLoader`.
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

## Configuration

Important application settings include:

| Setting | Description |
|---|---|
| `MODEL_NAME` | Groq model configuration |
| `TOP_K` | Number of RBI passages retrieved |
| `CHUNK_CHARS` | Context length per retrieved passage |
| `HISTORY_TURNS` | Conversational memory |
| `DATA_START` | Historical market data start date |
| `CACHE_TTL` | Yahoo Finance cache duration |
| `SHOW_SOURCES_BOX` | Display retrieved RBI passages |
| `SHOW_PAGE_NUMBERS` | Display PDF page references |

Additional token budgets and session limits may be configured for demo usage.

## Deployment

The application can be deployed using:

- [Streamlit Community Cloud](https://streamlit.io/cloud)
- [Hugging Face Spaces](https://huggingface.co/spaces)

For public deployment:

- Configure API keys using platform secrets.
- Never expose personal API credentials.
- Consider visitor rate limits and token budgets.
- Account for the memory requirements of the embedding model and FAISS index.

## Limitations

- Groq free-tier usage is subject to rate limits.
- Yahoo Finance data may contain missing or unavailable fields.
- Historical company financial statements may be limited.
- RBI PDF tables may not be extracted perfectly.
- Historical market news may not be available for older periods.
- Price movements cannot always be attributed to a single cause using market data alone.
- Historical market prices are based on available closing prices rather than real-time trading ticks.
- Important financial figures should be verified against official filings and RBI publications.

## Data Sources

- [Reserve Bank of India](https://www.rbi.org.in/)
- [Yahoo Finance](https://finance.yahoo.com/)
- [yfinance GitHub Repository](https://github.com/ranaroussi/yfinance)

## Disclaimer

This project is developed for educational, research, and demonstration purposes.

Generated responses are not financial advice, investment recommendations, or guarantees of market performance. Users should independently verify important information using official financial statements, RBI publications, and reliable market sources before making financial decisions.

## Author

Add your professional links here:

- **GitHub:** https://github.com/shravya184/
- **LinkedIn:** https://www.linkedin.com/in/shravya-dharshanala-197207339/

---

If you find this project interesting, consider starring the repository.
