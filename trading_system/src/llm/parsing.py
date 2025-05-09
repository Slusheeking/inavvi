"""
Parsing utilities for LLM responses.
"""
import json
import re
from typing import Dict, List, Optional, Union, Any

from src.utils.logging import setup_logger

# Set up logger
logger = setup_logger("llm_parsing")

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON data from text.
    
    Args:
        text: Text containing JSON data
        
    Returns:
        Extracted JSON data as dictionary, None if no valid JSON found
    """
    # First try to find JSON between triple backticks
    code_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    code_match = re.search(code_pattern, text)
    
    if code_match:
        json_str = code_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from code block, trying full text")
    
    # If no valid JSON in code blocks, try extracting from the entire text
    try:
        # Find the first opening brace
        start_idx = text.find('{')
        if start_idx == -1:
            logger.warning("No JSON object found in text")
            return None
        
        # Find the matching closing brace
        brace_count = 0
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    # We found the matching closing brace
                    json_str = text[start_idx:i+1]
                    return json.loads(json_str)
        
        logger.warning("No matching closing brace found for JSON object")
        return None
    except Exception as e:
        logger.error(f"Error extracting JSON from text: {e}")
        return None

def parse_trade_decision(response_content: str) -> Dict[str, Any]:
    """
    Parse trade decision from LLM response.
    
    Args:
        response_content: LLM response content
        
    Returns:
        Parsed trade decision
    """
    # Default decision (fail-safe)
    default_decision = {
        "decision": "no_trade",
        "position_size": 0.0,
        "confidence": 0.0,
        "reasoning": "Failed to parse response",
        "key_factors": []
    }
    
    # Extract JSON from response
    decision = extract_json_from_text(response_content)
    
    if not decision:
        logger.error("Failed to extract JSON from trade decision response")
        return default_decision
    
    # Validate and normalize decision
    if "decision" not in decision:
        logger.error("Missing 'decision' in trade decision")
        return default_decision
    
    # Normalize decision
    decision["decision"] = decision["decision"].lower()
    if decision["decision"] not in ["trade", "no_trade"]:
        logger.warning(f"Invalid decision value: {decision['decision']}, defaulting to 'no_trade'")
        decision["decision"] = "no_trade"
    
    # Ensure position_size is valid
    if "position_size" not in decision:
        decision["position_size"] = 0.0
    else:
        try:
            decision["position_size"] = float(decision["position_size"])
            if decision["position_size"] < 0.0:
                decision["position_size"] = 0.0
            elif decision["position_size"] > 1.0:
                decision["position_size"] = 1.0
        except (ValueError, TypeError):
            logger.warning("Invalid position_size value, defaulting to 0.0")
            decision["position_size"] = 0.0
    
    # If decision is no_trade, ensure position_size is 0
    if decision["decision"] == "no_trade":
        decision["position_size"] = 0.0
    
    # Ensure confidence is valid
    if "confidence" not in decision:
        decision["confidence"] = 0.5
    else:
        try:
            decision["confidence"] = float(decision["confidence"])
            if decision["confidence"] < 0.0:
                decision["confidence"] = 0.0
            elif decision["confidence"] > 1.0:
                decision["confidence"] = 1.0
        except (ValueError, TypeError):
            logger.warning("Invalid confidence value, defaulting to 0.5")
            decision["confidence"] = 0.5
    
    # Ensure reasoning is valid
    if "reasoning" not in decision or not decision["reasoning"]:
        decision["reasoning"] = "No reasoning provided"
    
    # Ensure key_factors is valid
    if "key_factors" not in decision or not isinstance(decision["key_factors"], list):
        decision["key_factors"] = []
    
    # Add the raw response for debugging
    decision["raw_response"] = response_content
    
    return decision

def parse_exit_decision(response_content: str) -> Dict[str, Any]:
    """
    Parse exit decision from LLM response.
    
    Args:
        response_content: LLM response content
        
    Returns:
        Parsed exit decision
    """
    # Default decision (fail-safe)
    default_decision = {
        "decision": "hold",
        "exit_size": 0.0,
        "confidence": 0.0,
        "reasoning": "Failed to parse response",
        "key_factors": []
    }
    
    # Extract JSON from response
    decision = extract_json_from_text(response_content)
    
    if not decision:
        logger.error("Failed to extract JSON from exit decision response")
        return default_decision
    
    # Validate and normalize decision
    if "decision" not in decision:
        logger.error("Missing 'decision' in exit decision")
        return default_decision
    
    # Normalize decision
    decision["decision"] = decision["decision"].lower()
    if decision["decision"] not in ["exit", "hold"]:
        logger.warning(f"Invalid decision value: {decision['decision']}, defaulting to 'hold'")
        decision["decision"] = "hold"
    
    # Ensure exit_size is valid
    if "exit_size" not in decision:
        decision["exit_size"] = 0.0
    else:
        try:
            decision["exit_size"] = float(decision["exit_size"])
            if decision["exit_size"] < 0.0:
                decision["exit_size"] = 0.0
            elif decision["exit_size"] > 1.0:
                decision["exit_size"] = 1.0
        except (ValueError, TypeError):
            logger.warning("Invalid exit_size value, defaulting to 0.0")
            decision["exit_size"] = 0.0
    
    # If decision is hold, ensure exit_size is 0
    if decision["decision"] == "hold":
        decision["exit_size"] = 0.0
    
    # Ensure confidence is valid
    if "confidence" not in decision:
        decision["confidence"] = 0.5
    else:
        try:
            decision["confidence"] = float(decision["confidence"])
            if decision["confidence"] < 0.0:
                decision["confidence"] = 0.0
            elif decision["confidence"] > 1.0:
                decision["confidence"] = 1.0
        except (ValueError, TypeError):
            logger.warning("Invalid confidence value, defaulting to 0.5")
            decision["confidence"] = 0.5
    
    # Ensure reasoning is valid
    if "reasoning" not in decision or not decision["reasoning"]:
        decision["reasoning"] = "No reasoning provided"
    
    # Ensure key_factors is valid
    if "key_factors" not in decision or not isinstance(decision["key_factors"], list):
        decision["key_factors"] = []
    
    # Add the raw response for debugging
    decision["raw_response"] = response_content
    
    return decision

def parse_market_analysis(response_content: str) -> Dict[str, Any]:
    """
    Parse market analysis from LLM response.
    
    Args:
        response_content: LLM response content
        
    Returns:
        Parsed market analysis
    """
    # Default analysis (fail-safe)
    default_analysis = {
        "market_regime": "uncertain",
        "confidence": 0.0,
        "reasoning": "Failed to parse response",
        "key_indicators": [],
        "trading_recommendation": "Proceed with caution"
    }
    
    # Extract JSON from response
    analysis = extract_json_from_text(response_content)
    
    if not analysis:
        logger.error("Failed to extract JSON from market analysis response")
        return default_analysis
    
    # Validate and normalize market_regime
    if "market_regime" not in analysis:
        logger.error("Missing 'market_regime' in market analysis")
        return default_analysis
    
    # Normalize market_regime
    analysis["market_regime"] = analysis["market_regime"].lower()
    valid_regimes = ["bullish", "bearish", "range_bound", "volatile", "uncertain"]
    if analysis["market_regime"] not in valid_regimes:
        logger.warning(f"Invalid market_regime value: {analysis['market_regime']}, defaulting to 'uncertain'")
        analysis["market_regime"] = "uncertain"
    
    # Ensure confidence is valid
    if "confidence" not in analysis:
        analysis["confidence"] = 0.5
    else:
        try:
            analysis["confidence"] = float(analysis["confidence"])
            if analysis["confidence"] < 0.0:
                analysis["confidence"] = 0.0
            elif analysis["confidence"] > 1.0:
                analysis["confidence"] = 1.0
        except (ValueError, TypeError):
            logger.warning("Invalid confidence value, defaulting to 0.5")
            analysis["confidence"] = 0.5
    
    # Ensure reasoning is valid
    if "reasoning" not in analysis or not analysis["reasoning"]:
        analysis["reasoning"] = "No reasoning provided"
    
    # Ensure key_indicators is valid
    if "key_indicators" not in analysis or not isinstance(analysis["key_indicators"], list):
        analysis["key_indicators"] = []
    
    # Ensure trading_recommendation is valid
    if "trading_recommendation" not in analysis or not analysis["trading_recommendation"]:
        analysis["trading_recommendation"] = "No recommendation provided"
    
    # Add the raw response for debugging
    analysis["raw_response"] = response_content
    
    return analysis