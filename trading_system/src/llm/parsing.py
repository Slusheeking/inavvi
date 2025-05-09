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

def parse_llm_response(response_content: str, task_type: str) -> Dict[str, Any]:
   """
   Parse LLM response based on task type.
   
   Args:
       response_content: Raw LLM response content
       task_type: Type of task/query sent to the LLM (e.g., "trade_decision", "exit_decision", "market_analysis")
       
   Returns:
       Parsed response as structured dictionary
   """
   logger.info(f"Parsing LLM response for task_type: {task_type}")
   
   try:
       if task_type == "trade_decision":
           return parse_trade_decision(response_content)
       elif task_type == "exit_decision":
           return parse_exit_decision(response_content)
       elif task_type == "market_analysis":
           return parse_market_analysis(response_content)
       else:
           logger.warning(f"Unknown task type: {task_type}, attempting to extract structured data")
           # For unknown task types, attempt to extract structured data
           structured_data = extract_structured_data(response_content)
           if not structured_data:
               return {
                   "error": "Unknown task type",
                   "task_type": task_type,
                   "raw_response": response_content
               }
           return structured_data
   except Exception as e:
       logger.error(f"Error parsing LLM response: {e}")
       return {
           "error": str(e),
           "task_type": task_type,
           "raw_response": response_content
       }

def extract_market_analysis(response_content: str) -> Dict[str, Any]:
   """
   Extract and structure market analysis insights from LLM response.
   
   This function focuses specifically on extracting market analysis data,
   potentially with a different structure than parse_market_analysis.
   
   Args:
       response_content: LLM response content containing market analysis
       
   Returns:
       Dictionary containing structured market analysis data
   """
   # Default structure for market analysis
   default_analysis = {
       "overall_sentiment": "neutral",
       "key_insights": [],
       "market_drivers": {},
       "risk_factors": []
   }
   
   # Try to extract structured data first
   structured_data = extract_json_from_text(response_content)
   
   if not structured_data:
       logger.warning("Could not extract structured market analysis, using text-based extraction")
       # If no structured data, try to extract insights from text
       
       # Extract overall sentiment using regex patterns
       sentiment_patterns = [
           (r"(?i)overall\s+(?:sentiment|outlook)(?:[:\s]+)(bullish|bearish|neutral)", 1),
           (r"(?i)(bullish|bearish|neutral)\s+(?:sentiment|outlook)", 1)
       ]
       
       for pattern, group in sentiment_patterns:
           match = re.search(pattern, response_content)
           if match:
               default_analysis["overall_sentiment"] = match.group(group).lower()
               break
       
       # Extract key insights - look for lists or bullet points
       insights_pattern = r"(?:key insights|key points|main takeaways)[:\n]+((?:[-•*]\s*[^\n]+\n?)+)"
       insights_match = re.search(insights_pattern, response_content, re.IGNORECASE)
       if insights_match:
           insights_text = insights_match.group(1)
           insights = re.findall(r"[-•*]\s*([^\n]+)", insights_text)
           default_analysis["key_insights"] = [insight.strip() for insight in insights]
       
       # Extract risk factors
       risks_pattern = r"(?:risk factors|risks|concerns)[:\n]+((?:[-•*]\s*[^\n]+\n?)+)"
       risks_match = re.search(risks_pattern, response_content, re.IGNORECASE)
       if risks_match:
           risks_text = risks_match.group(1)
           risks = re.findall(r"[-•*]\s*([^\n]+)", risks_text)
           default_analysis["risk_factors"] = [risk.strip() for risk in risks]
       
       return default_analysis
   
   # Process structured data
   analysis = {}
   
   # Transfer valid fields from structured data to analysis
   if "overall_sentiment" in structured_data:
       analysis["overall_sentiment"] = structured_data["overall_sentiment"].lower() if isinstance(structured_data["overall_sentiment"], str) else "neutral"
   else:
       analysis["overall_sentiment"] = "neutral"
   
   if "key_insights" in structured_data and isinstance(structured_data["key_insights"], list):
       analysis["key_insights"] = structured_data["key_insights"]
   else:
       analysis["key_insights"] = []
   
   if "market_drivers" in structured_data and isinstance(structured_data["market_drivers"], dict):
       analysis["market_drivers"] = structured_data["market_drivers"]
   else:
       analysis["market_drivers"] = {}
   
   if "risk_factors" in structured_data and isinstance(structured_data["risk_factors"], list):
       analysis["risk_factors"] = structured_data["risk_factors"]
   else:
       analysis["risk_factors"] = []
   
   # Add any additional fields from the structured data
   for key, value in structured_data.items():
       if key not in analysis:
           analysis[key] = value
   
   # Add raw response for debugging
   analysis["raw_response"] = response_content
   
   return analysis

def extract_sentiment_score(response_content: str) -> Optional[float]:
   """
   Extract a sentiment score from LLM response content.
   
   Args:
       response_content: LLM response content
       
   Returns:
       Sentiment score as float in range [-1.0, 1.0], or None if no score found
   """
   # Try to extract structured data first
   structured_data = extract_json_from_text(response_content)
   
   if structured_data and "sentiment_score" in structured_data:
       try:
           score = float(structured_data["sentiment_score"])
           # Ensure score is in [-1.0, 1.0] range
           if score < -1.0:
               score = -1.0
           elif score > 1.0:
               score = 1.0
           return score
       except (ValueError, TypeError):
           logger.warning("Invalid sentiment_score in structured data")
   
   # If no structured data or no sentiment_score field, try regex patterns
   patterns = [
       r"sentiment\s*(?:score|value|rating|index)?\s*[=:]\s*([-+]?\d*\.?\d+)",
       r"sentiment\s*(?:score|value|rating|index)?\s*(?:of|is)?\s*([-+]?\d*\.?\d+)",
       r"([-+]?\d*\.?\d+)\s*sentiment\s*(?:score|value|rating|index)?",
   ]
   
   for pattern in patterns:
       match = re.search(pattern, response_content, re.IGNORECASE)
       if match:
           try:
               score = float(match.group(1))
               # Check ranges and normalize if needed
               
               # If score is in [0, 100] range, normalize to [-1, 1]
               if 0 <= score <= 100:
                   normalized_score = (score / 50) - 1
                   return max(-1.0, min(1.0, normalized_score))
               
               # If score is in [0, 10] range, normalize to [-1, 1]
               elif 0 <= score <= 10:
                   normalized_score = (score / 5) - 1
                   return max(-1.0, min(1.0, normalized_score))
               
               # If score is in [0, 1] range, normalize to [-1, 1]
               elif 0 <= score <= 1:
                   normalized_score = score * 2 - 1
                   return normalized_score
               
               # If already in [-1, 1] range, return as is
               elif -1 <= score <= 1:
                   return score
               
               # Otherwise, limit to [-1, 1]
               else:
                   return max(-1.0, min(1.0, score))
               
           except ValueError:
               continue
   
   # If no numeric score found, look for sentiment keywords
   sentiment_keywords = {
       "extremely bearish": -1.0,
       "very bearish": -0.8,
       "bearish": -0.6,
       "somewhat bearish": -0.3,
       "slightly bearish": -0.2,
       "neutral": 0.0,
       "slightly bullish": 0.2,
       "somewhat bullish": 0.3,
       "bullish": 0.6,
       "very bullish": 0.8,
       "extremely bullish": 1.0
   }
   
   for keyword, score in sentiment_keywords.items():
       if keyword in response_content.lower():
           return score
   
   logger.warning("No sentiment score found in response")
   return None

def extract_trading_signals(response_content: str) -> List[str]:
   """
   Extract trading signals from LLM response content.
   
   Args:
       response_content: LLM response content
       
   Returns:
       List of trading signals as strings
   """
   # Try to extract structured data first
   structured_data = extract_json_from_text(response_content)
   
   if structured_data:
       # Check if there's a trading_signals field in the JSON
       if "trading_signals" in structured_data and isinstance(structured_data["trading_signals"], list):
           return structured_data["trading_signals"]
       
       # Check if there's a signals field in the JSON
       if "signals" in structured_data and isinstance(structured_data["signals"], list):
           return structured_data["signals"]
   
   # If no structured data or no appropriate field, try to extract from text
   signals = []
   
   # Pattern for common signal formats
   signal_patterns = [
       r"(?:trading signal|signal)[s]?[:\s]+(buy|sell|hold|long|short)",
       r"(?:recommendation|indicator)[s]?[:\s]+(buy|sell|hold|long|short)",
       r"(buy|sell|hold|long|short)\s+(?:signal|recommendation|indicator)",
   ]
   
   for pattern in signal_patterns:
       matches = re.findall(pattern, response_content, re.IGNORECASE)
       for match in matches:
           signal = match.lower().strip()
           if signal not in signals:
               signals.append(signal)
   
   # Try to find signals in list format (bullet points, numbered lists)
   list_pattern = r"(?:signals|recommendations|indicators)[:\n]+((?:[-•*\d\.]\s*[^\n]+\n?)+)"
   list_match = re.search(list_pattern, response_content, re.IGNORECASE)
   
   if list_match:
       list_text = list_match.group(1)
       list_items = re.findall(r"[-•*\d\.]\s*([^\n]+)", list_text)
       
       for item in list_items:
           # Check if the item contains known signal keywords
           signal_keywords = ["buy", "sell", "hold", "long", "short"]
           for keyword in signal_keywords:
               if keyword.lower() in item.lower():
                   signals.append(item.strip())
                   break
   
   return signals

def extract_structured_data(response_content: str) -> Dict[str, Any]:
   """
   Extract structured data from LLM response with validation and error handling.
   
   This function extends extract_json_from_text with specific validation
   and normalization for common financial data structures.
   
   Args:
       response_content: LLM response content
       
   Returns:
       Dictionary containing structured data
   """
   # First try to extract any JSON data from the text
   data = extract_json_from_text(response_content)
   
   if not data:
       logger.warning("No structured data found in response")
       # Try to create a basic structure from the text
       data = {
           "raw_text": response_content,
           "extracted_data": False
       }
       
       # Try to extract key-value pairs in format "key: value" from the text
       kv_pattern = r"([A-Za-z_][A-Za-z0-9_\s]*?):\s*([^\n:]+)"
       kv_matches = re.findall(kv_pattern, response_content)
       
       if kv_matches:
           extracted = {}
           for key, value in kv_matches:
               key = key.strip().lower().replace(" ", "_")
               value = value.strip()
               
               # Try to convert numeric values
               if re.match(r"^[-+]?\d*\.?\d+$", value):
                   try:
                       if "." in value:
                           extracted[key] = float(value)
                       else:
                           extracted[key] = int(value)
                   except ValueError:
                       extracted[key] = value
               # Try to convert boolean values
               elif value.lower() in ["true", "false"]:
                   extracted[key] = value.lower() == "true"
               else:
                   extracted[key] = value
           
           if extracted:
               data["extracted_key_values"] = extracted
               data["extracted_data"] = True
   
   # Process specific financial data fields for validation and normalization
   if data:
       # Normalize sentiment values
       if "sentiment" in data and isinstance(data["sentiment"], str):
           sentiment_mapping = {
               "very bearish": "very_bearish",
               "bearish": "bearish",
               "slightly bearish": "slightly_bearish",
               "neutral": "neutral",
               "slightly bullish": "slightly_bullish",
               "bullish": "bullish",
               "very bullish": "very_bullish"
           }
           
           for text, normalized in sentiment_mapping.items():
               if data["sentiment"].lower() == text:
                   data["sentiment"] = normalized
                   break
       
       # Normalize and validate confidence values
       if "confidence" in data:
           try:
               confidence = float(data["confidence"])
               # Ensure confidence is in [0, 1] range
               data["confidence"] = max(0.0, min(1.0, confidence))
           except (ValueError, TypeError):
               if isinstance(data["confidence"], str):
                   # Try to convert textual confidence to numeric
                   confidence_mapping = {
                       "very low": 0.1,
                       "low": 0.3,
                       "medium": 0.5,
                       "high": 0.7,
                       "very high": 0.9
                   }
                   
                   for text, value in confidence_mapping.items():
                       if text in data["confidence"].lower():
                           data["confidence"] = value
                           break
       
       # Ensure any price targets are numeric
       if "price_target" in data:
           try:
               data["price_target"] = float(data["price_target"])
           except (ValueError, TypeError):
               # If conversion fails, keep the original value
               pass
   
   return data