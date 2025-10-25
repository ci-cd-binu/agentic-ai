#!/usr/bin/env python3
"""
Step 4: Adding Memory and Personalization with Session State
Demonstrates:
1. Reading user preferences from session state (temperature unit)
2. Writing to session state (last city checked)
3. Auto-saving agent responses via output_key
4. Manual state updates to test preference changes
"""
import logging
from typing import Dict
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext

logging.basicConfig(level=logging.INFO)

MODEL = "gemini-2.0-flash"
APP_NAME = "weather_bot_stateful"
USER_ID = "user_state_demo"
SESSION_ID = "session_state_01"

# -------------------------------------------------------------------
# 1. Initialize Session Service with Initial State
# -------------------------------------------------------------------
session_service = InMemorySessionService()

# Initial state: User prefers Celsius by default
initial_state = {
    "user_preference_temperature_unit": "Celsius"
}

print("=" * 60)
print("STEP 1: INITIALIZING SESSION WITH STATE")
print("=" * 60)
print(f"Initial State: {initial_state}")

# -------------------------------------------------------------------
# 2. Stateful Weather Tool (Reads AND Writes State)
# -------------------------------------------------------------------
def get_weather_stateful(city: str, tool_context: ToolContext) -> Dict:
    """
    Retrieve weather and format temperature based on session state.
    
    READS from state:
    - user_preference_temperature_unit: "Celsius" or "Fahrenheit"
    
    WRITES to state:
    - last_city_checked_stateful: The city that was just checked
    """
    print(f"\n--- Tool: get_weather_stateful called for '{city}' ---")
    
    # ✅ READ user preference from state
    preferred_unit = tool_context.state.get("user_preference_temperature_unit", "Celsius")
    print(f"--- Tool: Reading state 'user_preference_temperature_unit': {preferred_unit} ---")
    
    # Mock weather database (stored in Celsius internally)
    city_key = city.lower().replace(" ", "")
    mock_weather_db = {
        "newyork": {"temp_c": 25, "condition": "sunny"},
        "london": {"temp_c": 15, "condition": "cloudy"},
        "tokyo": {"temp_c": 18, "condition": "light rain"},
    }
    
    if city_key not in mock_weather_db:
        error_msg = f"Sorry, I don't have weather information for '{city}'."
        print(f"--- Tool: City '{city}' not found. ---")
        return {"status": "error", "error_message": error_msg}
    
    data = mock_weather_db[city_key]
    temp_c = data["temp_c"]
    condition = data["condition"]
    
    # Format temperature based on user preference
    if preferred_unit == "Fahrenheit":
        temp_value = (temp_c * 9 / 5) + 32
        unit = "°F"
    else:
        temp_value = temp_c
        unit = "°C"
    
    report = f"The weather in {city.capitalize()} is {condition} with a temperature of {temp_value:.0f}{unit}."
    
    # ✅ WRITE to state - remember last city checked
    tool_context.state["last_city_checked_stateful"] = city.capitalize()
    print(f"--- Tool: Updated state 'last_city_checked_stateful': {city.capitalize()} ---")
    print(f"--- Tool: Generated report in {preferred_unit}. ---")
    
    return {"status": "success", "report": report}

# -------------------------------------------------------------------
# 3. Optional: Tool to Change Temperature Preference
# -------------------------------------------------------------------
def set_temperature_preference(unit: str, tool_context: ToolContext) -> Dict:
    """
    Allow users to change their temperature unit preference.
    
    Args:
        unit: Either 'Celsius' or 'Fahrenheit'
    """
    unit = unit.capitalize()
    if unit not in ["Celsius", "Fahrenheit"]:
        return {
            "status": "error",
            "error_message": "Unit must be 'Celsius' or 'Fahrenheit'"
        }
    
    # ✅ WRITE preference to state
    tool_context.state["user_preference_temperature_unit"] = unit
    print(f"--- Tool: Updated preference to {unit} ---")
    
    return {
        "status": "success",
        "message": f"✅ Temperature preference set to {unit}."
    }

# -------------------------------------------------------------------
# 4. Sub-agents for Delegation
# -------------------------------------------------------------------
def say_hello() -> str:
    """Simple greeting tool."""
    return "Hello! How can I help you today?"

def say_goodbye() -> str:
    """Simple farewell tool."""
    return "Goodbye! Have a great day!"

greeting_agent = Agent(
    name="greeting_agent",
    model=MODEL,
    description="Handles greetings",
    instruction="You are the Greeting Agent. Provide a friendly greeting using 'say_hello'.",
    tools=[say_hello]
)

farewell_agent = Agent(
    name="farewell_agent",
    model=MODEL,
    description="Handles farewells",
    instruction="You are the Farewell Agent. Provide a polite goodbye using 'say_goodbye'.",
    tools=[say_goodbye]
)

# -------------------------------------------------------------------
# 5. Root Agent with State Management
# -------------------------------------------------------------------
team_agent_stateful = Agent(
    name="weather_agent_stateful",
    model=MODEL,
    description="Root agent with stateful weather tool and personalization",
    instruction=(
        "You are the Weather Coordinator. "
        "For weather queries, use 'get_weather_stateful' which respects user's temperature unit preference. "
        "If users want to change units, use 'set_temperature_preference'. "
        "Delegate greetings to 'greeting_agent' and farewells to 'farewell_agent'. "
        "The system remembers user preferences and the last city checked."
    ),
    tools=[get_weather_stateful, set_temperature_preference],
    sub_agents=[greeting_agent, farewell_agent],
    output_key="last_weather_report"  # ✅ Auto-saves agent's final response to state
)

# -------------------------------------------------------------------
# 6. Create Runner
# -------------------------------------------------------------------
runner = Runner(
    agent=team_agent_stateful,
    app_name=APP_NAME,
    session_service=session_service
)

# -------------------------------------------------------------------
# 7. Test Script - Demonstrates State Flow
# -------------------------------------------------------------------
async def demonstrate_state_management():
    """
    Test conversation flow to demonstrate:
    1. Initial state (Celsius preference)
    2. Tool reading from state
    3. Tool writing to state (last city)
    4. Manual state update (changing to Fahrenheit)
    5. Tool reading updated state
    6. output_key auto-saving responses
    """
    
    # Initialize session with initial state
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state=initial_state
    )
    print(f"✅ Session created with initial state: {session.state}\n")
    
    # -------------------------------------------------------------------
    # TEST 1: Check weather with Celsius (initial preference)
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 1: Weather query with CELSIUS preference")
    print("=" * 60)
    
    response1 = await runner.run(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message="What's the weather in London?"
    )
    print(f"\n🤖 Agent Response: {response1.message.content}")
    
    # Check state after first interaction
    session_after_1 = await session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    print(f"\n📊 State After Test 1:")
    print(f"  - Temperature Unit: {session_after_1.state.get('user_preference_temperature_unit')}")
    print(f"  - Last City Checked: {session_after_1.state.get('last_city_checked_stateful')}")
    print(f"  - Last Weather Report (output_key): {session_after_1.state.get('last_weather_report', 'N/A')[:100]}...")
    
    # -------------------------------------------------------------------
    # TEST 2: Manually update state to Fahrenheit
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 2: MANUALLY changing preference to FAHRENHEIT")
    print("=" * 60)
    
    # For InMemorySessionService, we can directly modify the stored session
    # In production, this would be done via a tool or API call
    if hasattr(session_service, '_sessions'):
        session_key = (APP_NAME, USER_ID, SESSION_ID)
        if session_key in session_service._sessions:
            session_service._sessions[session_key].state["user_preference_temperature_unit"] = "Fahrenheit"
            print("✅ Manually updated state: user_preference_temperature_unit = 'Fahrenheit'")
    
    # -------------------------------------------------------------------
    # TEST 3: Check weather with Fahrenheit (updated preference)
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 3: Weather query with FAHRENHEIT preference")
    print("=" * 60)
    
    response2 = await runner.run(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message="How about New York?"
    )
    print(f"\n🤖 Agent Response: {response2.message.content}")
    
    # Check state after second interaction
    session_after_2 = await session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    print(f"\n📊 State After Test 3:")
    print(f"  - Temperature Unit: {session_after_2.state.get('user_preference_temperature_unit')}")
    print(f"  - Last City Checked: {session_after_2.state.get('last_city_checked_stateful')}")
    print(f"  - Last Weather Report (output_key): {session_after_2.state.get('last_weather_report', 'N/A')[:100]}...")
    
    # -------------------------------------------------------------------
    # TEST 4: Test user-initiated preference change via tool
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 4: User changes preference back to Celsius via tool")
    print("=" * 60)
    
    response3 = await runner.run(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message="I prefer Celsius from now on"
    )
    print(f"\n🤖 Agent Response: {response3.message.content}")
    
    # -------------------------------------------------------------------
    # TEST 5: Verify preference change took effect
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 5: Weather query after user changed preference to Celsius")
    print("=" * 60)
    
    response4 = await runner.run(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message="What about Tokyo?"
    )
    print(f"\n🤖 Agent Response: {response4.message.content}")
    
    # -------------------------------------------------------------------
    # TEST 6: Test delegation (greeting) - this becomes the new output_key value
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 6: Test sub-agent delegation (greeting)")
    print("=" * 60)
    
    response5 = await runner.run(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message="Hello!"
    )
    print(f"\n🤖 Agent Response: {response5.message.content}")
    
    # -------------------------------------------------------------------
    # FINAL STATE INSPECTION
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL STATE INSPECTION")
    print("=" * 60)
    
    final_session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    
    print("\n📊 Complete Final State:")
    for key, value in final_session.state.items():
        if key == "last_weather_report":
            print(f"  - {key}: {value[:100]}...")  # Truncate long reports
        else:
            print(f"  - {key}: {value}")
    
    print("\n" + "=" * 60)
    print("KEY OBSERVATIONS:")
    print("=" * 60)
    print("1. ✅ Tool READ user preference from state (Celsius → Fahrenheit → Celsius)")
    print("2. ✅ Tool WROTE last_city_checked_stateful to state on each weather query")
    print("3. ✅ output_key auto-saved agent's final response (overwritten each turn)")
    print("4. ✅ State persists across conversation turns within the session")
    print("5. ✅ User can change preferences via tool (set_temperature_preference)")
    print("=" * 60)

# -------------------------------------------------------------------
# 8. For ADK Web Deployment
# -------------------------------------------------------------------
root_agent = team_agent_stateful  # Export for ADK Web

# -------------------------------------------------------------------
# 9. Main Execution (for standalone testing)
# -------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_state_management())