#!/usr/bin/env python3
"""
ADK Query Cache Implementation - Fixed Version
- Adds independent sub-agent instances for each root agent.
- Allows dynamic selection between 'simple' and 'semantic' caching.
"""

import os
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext

logging.basicConfig(level=logging.INFO)

MODEL = "gemini-2.0-flash"
APP_NAME = "weather_bot_cached"
CACHE_SIZE = 5
CACHE_TTL_MINUTES = 30

# =============================================================================
# UTILS
# =============================================================================

def normalize_query(query: str) -> str:
    """Normalize query for matching (lowercase, strip spaces)."""
    return query.lower().strip()

# =============================================================================
# SIMPLE CACHE IMPLEMENTATION
# =============================================================================

def get_from_cache_simple(query: str, tool_context: ToolContext) -> Optional[str]:
    cache = tool_context.state.get("query_cache", [])
    normalized_query = normalize_query(query)
    for entry in cache:
        if normalize_query(entry["query"]) == normalized_query:
            cached_time = datetime.fromisoformat(entry["timestamp"])
            if datetime.now() - cached_time < timedelta(minutes=CACHE_TTL_MINUTES):
                print(f"✅ CACHE HIT: Found cached response for '{query}'")
                return entry["response"]
            else:
                print(f"⏰ CACHE EXPIRED for '{query}'")
    print(f"❌ CACHE MISS: '{query}' not in cache")
    return None

def add_to_cache_simple(query: str, response: str, tool_context: ToolContext):
    cache = tool_context.state.get("query_cache", [])
    cache = [e for e in cache if normalize_query(e["query"]) != normalize_query(query)]
    new_entry = {"query": query, "response": response, "timestamp": datetime.now().isoformat()}
    cache.append(new_entry)
    if len(cache) > CACHE_SIZE:
        cache = cache[-CACHE_SIZE:]
    tool_context.state["query_cache"] = cache
    print(f"💾 Added '{query}' to simple cache (size={len(cache)})")

def get_weather_cached_simple(city: str, tool_context: ToolContext) -> Dict:
    """Weather tool with simple caching."""
    query_key = f"weather in {city}"
    cached_response = get_from_cache_simple(query_key, tool_context)
    if cached_response:
        return {"status": "success", "report": cached_response, "from_cache": True}

    # Mock weather DB
    weather_db = {
        "newyork": {"temp_c": 25, "condition": "sunny"},
        "london": {"temp_c": 15, "condition": "cloudy"},
        "tokyo": {"temp_c": 18, "condition": "light rain"},
    }

    city_key = city.lower().replace(" ", "")
    if city_key not in weather_db:
        return {"status": "error", "error_message": f"No info for '{city}'."}

    preferred_unit = tool_context.state.get("user_preference_temperature_unit", "Celsius")
    temp_c = weather_db[city_key]["temp_c"]
    if preferred_unit == "Fahrenheit":
        temp_value, unit = (temp_c * 9 / 5) + 32, "°F"
    else:
        temp_value, unit = temp_c, "°C"

    report = f"The weather in {city.capitalize()} is {weather_db[city_key]['condition']} with {temp_value:.0f}{unit}."
    add_to_cache_simple(query_key, report, tool_context)
    tool_context.state["last_city_checked"] = city.capitalize()
    return {"status": "success", "report": report, "from_cache": False}

# =============================================================================
# SEMANTIC CACHE IMPLEMENTATION
# =============================================================================

def get_query_similarity(query1: str, query2: str) -> float:
    words1, words2 = set(normalize_query(query1).split()), set(normalize_query(query2).split())
    inter, uni = len(words1 & words2), len(words1 | words2)
    return inter / uni if uni else 0.0

def get_from_cache_semantic(query: str, tool_context: ToolContext, threshold: float = 0.6) -> Optional[str]:
    cache = tool_context.state.get("query_cache_semantic", [])
    best_match = None
    best_sim = 0.0
    for entry in cache:
        sim = get_query_similarity(query, entry["query"])
        if sim > best_sim and sim >= threshold:
            cached_time = datetime.fromisoformat(entry["timestamp"])
            if datetime.now() - cached_time < timedelta(minutes=CACHE_TTL_MINUTES):
                best_sim, best_match = sim, entry
    if best_match:
        print(f"✅ SEMANTIC CACHE HIT: '{query}' ≈ '{best_match['query']}' ({best_sim:.2f})")
        return best_match["response"]
    print(f"❌ SEMANTIC CACHE MISS: '{query}' not found")
    return None

def add_to_cache_semantic(query: str, response: str, tool_context: ToolContext):
    cache = tool_context.state.get("query_cache_semantic", [])
    for i, entry in enumerate(cache):
        if get_query_similarity(query, entry["query"]) > 0.9:
            cache[i] = {"query": query, "response": response, "timestamp": datetime.now().isoformat()}
            tool_context.state["query_cache_semantic"] = cache
            return
    cache.append({"query": query, "response": response, "timestamp": datetime.now().isoformat()})
    if len(cache) > CACHE_SIZE:
        cache = cache[-CACHE_SIZE:]
    tool_context.state["query_cache_semantic"] = cache
    print(f"💾 Added '{query}' to semantic cache (size={len(cache)})")

def get_weather_cached_semantic(city: str, tool_context: ToolContext) -> Dict:
    """Weather tool with semantic caching."""
    query_key = f"weather in {city}"
    cached_response = get_from_cache_semantic(query_key, tool_context)
    if cached_response:
        return {"status": "success", "report": cached_response + " (from smart cache)", "from_cache": True}

    # Same weather logic as simple version
    weather_db = {
        "newyork": {"temp_c": 25, "condition": "sunny"},
        "london": {"temp_c": 15, "condition": "cloudy"},
        "tokyo": {"temp_c": 18, "condition": "light rain"},
    }
    city_key = city.lower().replace(" ", "")
    if city_key not in weather_db:
        return {"status": "error", "error_message": f"No info for '{city}'."}

    preferred_unit = tool_context.state.get("user_preference_temperature_unit", "Celsius")
    temp_c = weather_db[city_key]["temp_c"]
    temp_value, unit = ((temp_c * 9 / 5) + 32, "°F") if preferred_unit == "Fahrenheit" else (temp_c, "°C")

    report = f"The weather in {city.capitalize()} is {weather_db[city_key]['condition']} with {temp_value:.0f}{unit}."
    add_to_cache_semantic(query_key, report, tool_context)
    tool_context.state["last_city_checked"] = city.capitalize()
    return {"status": "success", "report": report, "from_cache": False}

# =============================================================================
# CACHE MANAGEMENT TOOLS
# =============================================================================

def view_cache(tool_context: ToolContext) -> Dict:
    simple = tool_context.state.get("query_cache", [])
    semantic = tool_context.state.get("query_cache_semantic", [])
    return {"status": "success", "cache_info": {"simple": len(simple), "semantic": len(semantic)}}

def clear_cache(tool_context: ToolContext) -> Dict:
    tool_context.state["query_cache"] = []
    tool_context.state["query_cache_semantic"] = []
    return {"status": "success", "message": "Cache cleared"}

# =============================================================================
# PREFERENCE TOOL
# =============================================================================

def set_temperature_preference(unit: str, tool_context: ToolContext) -> Dict:
    unit = unit.capitalize()
    if unit not in ["Celsius", "Fahrenheit"]:
        return {"status": "error", "error_message": "Must be Celsius or Fahrenheit"}
    tool_context.state["user_preference_temperature_unit"] = unit
    return {"status": "success", "message": f"Preference set to {unit}"}

# =============================================================================
# SUB-AGENTS FACTORY
# =============================================================================

def say_hello() -> str:
    return "Hello! I can check weather for you and remember recent queries."

def say_goodbye() -> str:
    return "Goodbye! Your preferences and history are saved."

def make_greeting_agents(suffix: str):
    greeting = Agent(
        name=f"greeting_agent_{suffix}",
        model=MODEL,
        description="Handles greetings",
        instruction="Respond to user greetings warmly.",
        tools=[say_hello],
    )
    farewell = Agent(
        name=f"farewell_agent_{suffix}",
        model=MODEL,
        description="Handles farewells",
        instruction="Respond politely to user farewells.",
        tools=[say_goodbye],
    )
    return greeting, farewell

# =============================================================================
# ROOT AGENTS (SIMPLE + SEMANTIC)
# =============================================================================

greeting_simple, farewell_simple = make_greeting_agents("simple")
greeting_semantic, farewell_semantic = make_greeting_agents("semantic")

weather_agent_simple = Agent(
    name="weather_agent_cached_simple",
    model=MODEL,
    description="Weather agent with exact match caching",
    instruction=(
        "Use get_weather_cached_simple to fetch weather info. "
        "Use caching, and delegate greetings/farewells as needed."
    ),
    tools=[get_weather_cached_simple, set_temperature_preference, view_cache, clear_cache],
    sub_agents=[greeting_simple, farewell_simple],
    output_key="last_weather_report",
)

weather_agent_semantic = Agent(
    name="weather_agent_cached_semantic",
    model=MODEL,
    description="Weather agent with semantic caching",
    instruction=(
        "Use get_weather_cached_semantic to answer flexibly matched weather queries. "
        "Delegate greetings/farewells as needed."
    ),
    tools=[get_weather_cached_semantic, set_temperature_preference, view_cache, clear_cache],
    sub_agents=[greeting_semantic, farewell_semantic],
    output_key="last_weather_report",
)

# =============================================================================
# ROOT AGENT SELECTION (DYNAMIC)
# =============================================================================

CACHE_MODE = os.getenv("CACHE_MODE", "semantic").lower()
if CACHE_MODE == "simple":
    root_agent = weather_agent_simple
else:
    root_agent = weather_agent_semantic

# =============================================================================
# OPTIONAL TEST HARNESS
# =============================================================================

async def test_agent():
    session_service = InMemorySessionService()
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)
    user_id, session_id = "user1", "s1"
    await session_service.create_session(APP_NAME, user_id, session_id)
    for msg in ["Hi", "What's the weather in London?", "London weather?", "Bye"]:
        result = await runner.run(user_id, session_id, msg)
        print(result.message.content)

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_agent())
