# System Architecture

```mermaid
flowchart TD
    subgraph "Data Processing"
        DATA_APIS[Data Feed]
        ALPHA_VANTAGE[Alpha Vantage]
        POLYGON[Polygon]
        YAHOO[Yahoo Finance]
        DATA_CACHE[(Redis Cache)]
        
        subgraph "AI/ML Models"
            ML_ANALYSIS[Market Analysis]
            ML_MONITOR[Position Monitor]
            SENTIMENT[Sentiment Analysis]
        end
        REDIS[(Redis Store)]
    end
    
    subgraph "Main LLM"
        MLLM[Main LLM]
        ANALYSIS[Trade Analysis]
        SIZING[Position Sizing]
    end
    
    subgraph "Trade LLM"
        TLLM[Trade LLM]
    end
    
    %% Data Sources
    ALPHA_VANTAGE --> DATA_APIS
    POLYGON --> DATA_APIS
    YAHOO --> DATA_APIS
    
    %% Data Flow
    DATA_APIS -->|Raw Data| DATA_CACHE
    DATA_CACHE -->|Market Data| ML_ANALYSIS
    DATA_CACHE -->|Position Data| ML_MONITOR
    DATA_CACHE -->|News & Sentiment| SENTIMENT
    
    %% ML to Redis
    ML_ANALYSIS -->|Processed Market Data| REDIS
    ML_MONITOR -->|Position Status| REDIS
    SENTIMENT -->|Sentiment Scores| REDIS
    
    %% Direct Signal Flow to Trade LLM
    ML_MONITOR -->|Position Status & Signals| TLLM
    
    %% Main LLM Flow
    REDIS -->|Market Data| MLLM
    MLLM --> ANALYSIS
    ANALYSIS --> SIZING
    SIZING -->|Trade Order| ALPACA[Alpaca Broker]
    ALPACA -->|Order Status| REDIS
    
    %% Trade LLM Flow
    TLLM -->|Exit Order| ALPACA
    
    %% Feedback Loop
    ALPACA -->|Position Created| REDIS