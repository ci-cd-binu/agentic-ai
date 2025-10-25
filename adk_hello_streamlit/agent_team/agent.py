#!/usr/bin/env python3
"""
Step 3: Building an Agent Team – Delegation for Greetings & Farewells
(using latest ADK API with sub_agents)
"""

import asyncio
import logging
from typing import Dict

from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

# -------------------------------------------------------------------
# Basic setup
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
MODEL = "gemini-2.0-flash"

# -------------------------------------------------------------------
# Shared tool: simple get_weather mock
# -------------------------------------------------------------------
def get_weather(city: str) -> Dict:
    """Mock weather lookup."""
    city_key = city.lower().replace(" ", "")
    weather_db = {
        "newyork": {"status": "success", "report": "Sunny, 25 °C"},
        "london": {"status": "success", "report": "Cloudy, 15 °C"},
        "tokyo": {"status": "success", "report": "Light rain, 18 °C"},
    }
    if city_key in weather_db:
        return weather_db[city_key]
    return {"status": "error", "error_message": f"No info for '{city}'."}

# -------------------------------------------------------------------
# Sub-agents
# -------------------------------------------------------------------
greeting_agent = Agent(
    name="greeting_agent",
    model=MODEL,
    description="Handles user greetings",
    instruction="Respond warmly to greetings such as hello, hi, good morning, etc.",
)

farewell_agent = Agent(
    name="farewell_agent",
    model=MODEL,
    description="Handles user farewells",
    instruction="Respond politely and naturally to farewells such as bye or see you later.",
)

weather_agent = Agent(
    name="weather_agent",
    model=MODEL,
    description="Provides weather information",
    instruction="Use get_weather(city) to answer weather questions clearly.",
    tools=[get_weather],
)

# -------------------------------------------------------------------
# Team‐controller (root agent) using sub_agents
# -------------------------------------------------------------------
team_agent = Agent(
    name="team_controller",
    model=MODEL,
    description="Routes incoming user messages to the appropriate specialized agent.",
    instruction=(
        "You are the team coordinator. Decide if the user's message is a greeting, "
        "a farewell, or a weather question. If it's a greeting, delegate to greeting_agent; "
        "if it's a farewell, delegate to farewell_agent; if it's about weather, delegate to weather_agent; "
        "otherwise answer directly your best."
    ),
    sub_agents=[greeting_agent, farewell_agent, weather_agent],
)

# -------------------------------------------------------------------
# Session + runner setup
# -------------------------------------------------------------------
APP_NAME = "agent_team_app"
USER_ID = "user_42"
SESSION_ID = "session_team_01"

session_service = InMemorySessionService()

async def make_runner() -> Runner:
    await session_service.create_session(APP_NAME, USER_ID, SESSION_ID)
    return Runner(agent=team_agent, app_name=APP_NAME, session_service=session_service)

# -------------------------------------------------------------------
# Helper to run user query through the team
# -------------------------------------------------------------------
async def ask_team(question: str, runner: Runner) -> str:
    print(f"\n>>> User: {question}")
    content = types.Content(role="user", parts=[types.Part(text=question)])
    final = ""
    async for event in runner.run_async(USER_ID, SESSION_ID, new_message=content):
        if event.is_final_response():
            if event.content and event.content.parts:
                final = event.content.parts[0].text
            else:
                final = "<no response text>"
            break
    print(f"<<< Agent team: {final}")
    return final

# -------------------------------------------------------------------
# Main demo
# -------------------------------------------------------------------
async def main_async():
    runner = await make_runner()
    await ask_team("Hi there!", runner)
    await ask_team("What’s the weather in London?", runner)
    await ask_team("Okay, bye!", runner)

def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("Interrupted.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

# Expose root agent so `adk web` can find it
root_agent = team_agent
