#!/bin/bash
# Trading system startup script

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}"
echo "========================================"
echo "    Day Trading System - A100 Edition   "
echo "========================================"
echo -e "${NC}"

# Check Python version
python_version=$(python3 --version)
echo -e "${BLUE}Using ${python_version}${NC}"

# Use the virtual environment located at trading_system/venv
VENV_PATH="venv"
if [ -d "$VENV_PATH" ]; then
    echo -e "${GREEN}Using virtual environment...${NC}"
    source $VENV_PATH/bin/activate
    echo -e "${GREEN}Virtual environment activated${NC}"
else
    echo -e "${RED}Error: Virtual environment not found at $VENV_PATH${NC}"
    echo "Please ensure the virtual environment is set up correctly"
    exit 1
fi

# Check for Redis server
redis_running=$(pgrep redis-server > /dev/null && echo "true" || echo "false")
if [ "$redis_running" = "false" ]; then
    echo -e "${YELLOW}Redis server not running. Starting Redis...${NC}"
    # Check if Redis is installed
    if command -v redis-server &> /dev/null; then
        redis-server --daemonize yes
        echo -e "${GREEN}Redis server started${NC}"
    else
        echo -e "${RED}Redis server not found. Please install Redis.${NC}"
        echo "On Ubuntu/Debian: sudo apt update && sudo apt install redis-server"
        echo "On macOS: brew install redis"
        exit 1
    fi
else
    echo -e "${GREEN}Redis server is already running${NC}"
fi

# Check environment variables
if [ ! -f ".env" ]; then
    echo -e "${RED}Error: .env file not found${NC}"
    echo "Please create an .env file with your API keys"
    exit 1
fi

# Check for required directories
mkdir -p logs data/raw data/processed models

# Parse command line arguments
COMMAND="loop"
if [ $# -gt 0 ]; then
    COMMAND=$1
fi

case $COMMAND in
    loop)
        echo -e "${BLUE}Starting trading system in main loop mode...${NC}"
        python trade.py
        ;;
    cycle)
        echo -e "${BLUE}Running a single trading cycle...${NC}"
        python trade.py --cycle
        ;;
    scan)
        echo -e "${BLUE}Running market scan...${NC}"
        python trade.py --scan
        ;;
    analyze)
        echo -e "${BLUE}Analyzing watchlist...${NC}"
        python trade.py --analyze
        ;;
    monitor)
        echo -e "${BLUE}Monitoring positions...${NC}"
        python trade.py --monitor
        ;;
    trade)
        echo -e "${BLUE}Making trade decisions...${NC}"
        python trade.py --trade
        ;;
    close)
        echo -e "${BLUE}Closing all positions...${NC}"
        python trade.py --close
        ;;
    train)
        echo -e "${BLUE}Training ML models...${NC}"
        python -m src.training.train_models
        ;;
    dashboard)
        echo -e "${BLUE}Starting dashboard...${NC}"
        python run.py --dashboard-only
        ;;
    api)
        echo -e "${BLUE}Starting API server...${NC}"
        python run.py --api-only
        ;;
    test)
        echo -e "${BLUE}Running tests...${NC}"
        python -m pytest -xvs tests/
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        echo "Available commands:"
        echo "  loop        - Start the trading system main loop"
        echo "  cycle       - Run a single trading cycle"
        echo "  scan        - Run market scan only"
        echo "  analyze     - Analyze watchlist only"
        echo "  monitor     - Monitor positions only"
        echo "  trade       - Make trade decisions only"
        echo "  close       - Close all positions"
        echo "  train       - Train ML models"
        echo "  dashboard   - Start dashboard only"
        echo "  api         - Start API server only"
        echo "  test        - Run tests"
        exit 1
        ;;
esac

# Deactivate virtual environment
deactivate
