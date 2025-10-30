#!/usr/bin/env python3
"""
GCP Cost Insight Agent (built with latest Google ADK)
FIXED VERSION - Corrected env var and column names
"""

import os
import logging
from typing import Dict

from dotenv import load_dotenv
from google.cloud import bigquery
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

# -------------------------------------------------------------------
# 🔧 Setup & Config
# -------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL = "gemini-2.0-flash"

# ✅ FIXED: Use GOOGLE_CLOUD_PROJECT (matches your .env)
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
DATASET = os.getenv("BILLING_DATASET")
TABLE = os.getenv("BILLING_TABLE")

# Validate environment variables
if not all([PROJECT_ID, DATASET, TABLE]):
    logger.error(f"Missing env vars - PROJECT_ID: {PROJECT_ID}, DATASET: {DATASET}, TABLE: {TABLE}")
    raise ValueError("Missing required environment variables")

logger.info(f"✅ Config loaded - Project: {PROJECT_ID}, Dataset: {DATASET}, Table: {TABLE}")

bq_client = bigquery.Client(project=PROJECT_ID)


# -------------------------------------------------------------------
# 🧩 Tool: BigQuery Cost Lookup
# -------------------------------------------------------------------
def get_cost_data(query: str) -> Dict:
    """Convert natural language to SQL and return summarized cost data."""
    logger.info(f"🔍 Processing query: {query}")
    q = query.lower()
    full_table = f"`{PROJECT_ID}.{DATASET}.{TABLE}`"

    if "yesterday" in q:
        sql = f"""
        SELECT service.description AS service_name, SUM(cost) AS total_cost
        FROM {full_table}
        WHERE DATE(usage_start_time) = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
        GROUP BY service.description
        ORDER BY total_cost DESC
        LIMIT 10
        """
    elif "last 7 days" in q or "past week" in q:
        sql = f"""
        SELECT service.description AS service_name, SUM(cost) AS total_cost
        FROM {full_table}
        WHERE usage_start_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
        GROUP BY service.description
        ORDER BY total_cost DESC
        LIMIT 10
        """
    elif "vertex ai" in q:
        sql = f"""
        SELECT DATE(usage_start_time) AS day, SUM(cost) AS total_cost
        FROM {full_table}
        WHERE service.description LIKE '%Vertex AI%'
        GROUP BY day
        ORDER BY day DESC
        LIMIT 10
        """
    else:
        sql = f"""
        SELECT service.description AS service_name, SUM(cost) AS total_cost
        FROM {full_table}
        WHERE usage_start_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
        GROUP BY service.description
        ORDER BY total_cost DESC
        LIMIT 10
        """

    try:
        logger.info(f"📊 Executing BigQuery query...")
        df = bq_client.query(sql).to_dataframe()
        logger.info(f"✅ Query returned {len(df)} rows")
        
        if df.empty:
            return {"status": "error", "message": "No billing data found for the specified period."}
        
        # ✅ FIXED: Use correct column names from SQL aliases
        if 'service_name' in df.columns:
            text_summary = "\n".join(
                f"{row['service_name']}: ${row['total_cost']:,.2f}" 
                for _, row in df.iterrows()
            )
        elif 'day' in df.columns:
            text_summary = "\n".join(
                f"{row['day']}: ${row['total_cost']:,.2f}" 
                for _, row in df.iterrows()
            )
        else:
            return {"status": "error", "message": f"Unexpected columns: {list(df.columns)}"}
        
        logger.info(f"✅ Generated summary with {len(text_summary)} characters")
        return {"status": "success", "report": text_summary}
        
    except Exception as e:
        logger.error(f"❌ Query failed: {e}")
        return {"status": "error", "message": str(e)}


# -------------------------------------------------------------------
# 🤖 Cost Agent
# -------------------------------------------------------------------
cost_agent = Agent(
    name="cost_agent",
    model=MODEL,
    description="Provides cost insights from GCP billing data.",
    instruction="Answer user questions about GCP costs using get_cost_data().",
    tools=[get_cost_data],
)


# -------------------------------------------------------------------
# 🧭 Root Controller Agent
# -------------------------------------------------------------------
root_agent = Agent(
    name="root_agent",
    model=MODEL,
    description="Main GCP FinOps assistant that routes cost-related queries.",
    instruction=(
        "You are a GCP FinOps AI assistant. "
        "If a user asks about costs, services, or billing, delegate to cost_agent. "
        "Otherwise, answer politely that you specialize in cost analysis."
    ),
    sub_agents=[cost_agent],
)

# For `adk web` discovery
if __name__ == "__main__":
    print("✅ Root agent ready: root_agent")
    print(f"📊 Using project: {PROJECT_ID}")
    print(f"📊 Using dataset: {DATASET}")
    print(f"📊 Using table: {TABLE}")