from google.adk.agents import Agent
import datetime
from zoneinfo import ZoneInfo

def get_weather(city: str) -> dict:
    """
    Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city for which to retrieve the weather report.

    Returns:
        dict: status and result or error msg.
    """
    if city.lower() == "new york":
        return {
            "status": "success",
            "report": (
                "The weather in New York is sunny with a temperature of 25 degrees"
                " Celsius (77 degrees Fahrenheit)."
            ),
        }
    else:
        return {
            "status": "error",
            "error_message": f"Weather data for {city} is not available.",
        }
    
root_agent = Agent(name="weather_time_agent",
                    model="gemini-2.0-flash",
                    description=(
                        "Agent to answer questions about weather in a city."
                    ),
                    instruction=(
                        "You are a helpful agent who can answer user questions about weather in a city."
                    ),
                    tools=[get_weather, ],
                )
