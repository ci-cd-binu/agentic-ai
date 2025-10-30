#!/usr/bin/env python3
"""
Standalone test script to debug FinOps agent issues
Run this BEFORE starting adk web to isolate the problem
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# -------------------------------------------------------------------
# TEST 1: Environment Variables
# -------------------------------------------------------------------
def test_environment():
    logger.info("=" * 60)
    logger.info("TEST 1: Environment Variables")
    logger.info("=" * 60)
    
    required_vars = {
        "GOOGLE_CLOUD_PROJECT": os.getenv("GOOGLE_CLOUD_PROJECT"),
        "BILLING_DATASET": os.getenv("BILLING_DATASET"),
        "BILLING_TABLE": os.getenv("BILLING_TABLE"),
        "GOOGLE_APPLICATION_CREDENTIALS": os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
    }
    
    all_ok = True
    for var_name, var_value in required_vars.items():
        if var_value:
            logger.info(f"✅ {var_name}: {var_value}")
        else:
            logger.error(f"❌ {var_name}: NOT SET")
            all_ok = False
    
    return all_ok


# -------------------------------------------------------------------
# TEST 2: BigQuery Connection
# -------------------------------------------------------------------
def test_bigquery():
    logger.info("=" * 60)
    logger.info("TEST 2: BigQuery Connection")
    logger.info("=" * 60)
    
    try:
        from google.cloud import bigquery
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        
        logger.info(f"Creating BigQuery client for project: {project_id}")
        client = bigquery.Client(project=project_id)
        
        # Try a simple query
        query = "SELECT 1 as test"
        logger.info(f"Testing with query: {query}")
        result = client.query(query).result()
        
        for row in result:
            logger.info(f"✅ Test query result: {row.test}")
        
        logger.info("✅ BigQuery connection successful")
        return True, client
        
    except Exception as e:
        logger.error(f"❌ BigQuery connection failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, None


# -------------------------------------------------------------------
# TEST 3: Check Billing Table Exists
# -------------------------------------------------------------------
def test_billing_table(client):
    logger.info("=" * 60)
    logger.info("TEST 3: Billing Table Access")
    logger.info("=" * 60)
    
    if not client:
        logger.error("❌ No BigQuery client available")
        return False
    
    try:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        dataset = os.getenv("BILLING_DATASET")
        table = os.getenv("BILLING_TABLE")
        full_table = f"`{project_id}.{dataset}.{table}`"
        
        query = f"SELECT COUNT(*) as row_count FROM {full_table} LIMIT 1"
        logger.info(f"Testing table access with: {query}")
        
        result = client.query(query).result()
        for row in result:
            logger.info(f"✅ Table accessible. Sample count: {row.row_count}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Cannot access billing table: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


# -------------------------------------------------------------------
# TEST 4: Google GenAI SDK
# -------------------------------------------------------------------
def test_genai_sdk():
    logger.info("=" * 60)
    logger.info("TEST 4: Google GenAI SDK")
    logger.info("=" * 60)
    
    try:
        import google.genai as genai
        logger.info(f"✅ google.genai version: {genai.__version__ if hasattr(genai, '__version__') else 'unknown'}")
        
        # Check if API key or credentials are set
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            logger.info("✅ GOOGLE_API_KEY is set")
        else:
            logger.warning("⚠️  GOOGLE_API_KEY not set (using ADC)")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Cannot import google.genai: {e}")
        return False


# -------------------------------------------------------------------
# TEST 5: ADK Import and Agent Creation
# -------------------------------------------------------------------
def test_adk_agent():
    logger.info("=" * 60)
    logger.info("TEST 5: ADK Agent Creation")
    logger.info("=" * 60)
    
    try:
        from google.adk.agents import Agent
        logger.info("✅ Successfully imported google.adk.agents")
        
        # Try creating a simple agent
        test_agent = Agent(
            name="test_agent",
            model="gemini-2.0-flash",
            description="Test agent",
            instruction="You are a test agent."
        )
        logger.info("✅ Successfully created test agent")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create agent: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


# -------------------------------------------------------------------
# TEST 6: Tool Function Test
# -------------------------------------------------------------------
def test_tool_function(client):
    logger.info("=" * 60)
    logger.info("TEST 6: Tool Function Direct Call")
    logger.info("=" * 60)
    
    if not client:
        logger.error("❌ No BigQuery client available")
        return False
    
    try:
        # Simulate the tool function
        from typing import Dict
        
        def get_cost_data_test(query: str) -> Dict:
            q = query.lower()
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
            dataset = os.getenv("BILLING_DATASET")
            table = os.getenv("BILLING_TABLE")
            full_table = f"`{project_id}.{dataset}.{table}`"
            
            sql = f"""
            SELECT service.description AS service_name, SUM(cost) AS total_cost
            FROM {full_table}
            WHERE usage_start_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
            GROUP BY service.description
            ORDER BY total_cost DESC
            LIMIT 5
            """
            
            logger.info(f"Executing test query:\n{sql}")
            df = client.query(sql).to_dataframe()
            logger.info(f"Returned {len(df)} rows")
            
            if df.empty:
                return {"status": "error", "message": "No data"}
            
            text_summary = "\n".join(
                f"{row['service_name']}: ${row['total_cost']:,.2f}" 
                for _, row in df.iterrows()
            )
            return {"status": "success", "report": text_summary}
        
        result = get_cost_data_test("show me costs")
        logger.info(f"✅ Tool function result: {result}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Tool function failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


# -------------------------------------------------------------------
# MAIN TEST RUNNER
# -------------------------------------------------------------------
def run_all_tests():
    logger.info("\n" + "=" * 60)
    logger.info("🔬 FINOPS AGENT DIAGNOSTIC TEST SUITE")
    logger.info("=" * 60 + "\n")
    
    results = {}
    
    # Test 1
    results['environment'] = test_environment()
    
    # Test 2
    results['bigquery'], bq_client = test_bigquery()
    
    # Test 3
    if results['bigquery']:
        results['billing_table'] = test_billing_table(bq_client)
    else:
        results['billing_table'] = False
    
    # Test 4
    results['genai_sdk'] = test_genai_sdk()
    
    # Test 5
    results['adk_agent'] = test_adk_agent()
    
    # Test 6
    if results['bigquery']:
        results['tool_function'] = test_tool_function(bq_client)
    else:
        results['tool_function'] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    logger.info("=" * 60)
    
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED - Agent should work")
    else:
        logger.error("⚠️  SOME TESTS FAILED - Check errors above")
    
    logger.info("=" * 60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)