#!/usr/bin/env python3
"""
Step 4: Adding Memory and Personalization with Session State
Version A — ADK Web (root_agent = team_agent_stateful)
"""

import logging
from typing import Dict
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext

logging.basicConfig(level=logging.INFO)
MODEL = "gemini-2.0-flash"

# -------------------------------------------------------------------
# 1. Stateful tool using ToolContext
# -------------------------------------------------------------------
def get_weather_stateful(city: str, tool_context: ToolContext) -> Dict:
    """Retrieve weather and format temperature based on session state."""
    preferred_unit = tool_context.state.get("user_preference_temperature_unit", "Celsius")
    city_key = city.lower().replace(" ", "")
    mock_weather_db = {
        "newyork": {"temp_c": 25, "condition": "sunny"},
        "london": {"temp_c": 15, "condition": "cloudy"},
        "tokyo": {"temp_c": 18, "condition": "light rain"},
    }

    if city_key not in mock_weather_db:
        return {"status": "error", "error_message": f"No info for '{city}'."}

    data = mock_weather_db[city_key]
    temp_c = data["temp_c"]
    if preferred_unit == "Fahrenheit":
        temp_value = (temp_c * 9 / 5) + 32
        unit = "°F"
    else:
        temp_value = temp_c
        unit = "°C"

    report = f"The weather in {city.capitalize()} is {data['condition']} with a temperature of {temp_value:.0f}{unit}."
    tool_context.state["last_city_checked_stateful"] = city.capitalize()
    return {"status": "success", "report": report}

# -------------------------------------------------------------------
# 2. Sub-agents for delegation
# -------------------------------------------------------------------
greeting_agent = Agent(
    name="greeting_agent",
    model=MODEL,
    description="Handles greetings",
    instruction="Respond warmly to greetings such as hi or hello."
)

farewell_agent = Agent(
    name="farewell_agent",
    model=MODEL,
    description="Handles farewells",
    instruction="Respond politely to bye or see you later."
)

# -------------------------------------------------------------------
# 3. Root agent with memory and output_key
# -------------------------------------------------------------------
team_agent_stateful = Agent(
    name="weather_agent_stateful",
    model=MODEL,
    description="Root agent with stateful weather tool and personalization",
    instruction=(
        "You are the coordinator. Use 'get_weather_stateful' for weather "
        "queries, delegate greetings/farewells appropriately, and remember "
        "user preferences stored in session state."
    ),
    tools=[get_weather_stateful],
    sub_agents=[greeting_agent, farewell_agent],
    output_key="last_weather_report"
)

# -------------------------------------------------------------------
# 4. Session initialization (for testing in standalone mode)
# -------------------------------------------------------------------
APP_NAME = "agent_team_stateful_app"
USER_ID = "user_stateful"
SESSION_ID = "session_state_01"

session_service = InMemorySessionService()

# Expose for ADK Web
root_agent = team_agent_stateful
