Here is a complete, step-by-step guide to building this pipeline on the GCP free tier, complete with synthetic data generation (with quality issues), BigQuery stored procedures for the Medallion architecture, and instructions on how to use Dataplex to showcase quality and lineage.

### 🚀 The POC Plan: A Serverless Medallion Pipeline with Dataplex

Our goal is to build a system that simulates data arriving, processes it through Bronze, Silver, and Gold layers, and uses Dataplex to govern and observe the entire process.

**Here is the architecture:**

1.  **Data Generation (Every 2 Mins):** A **Cloud Scheduler** job triggers a **Cloud Function** (Python).
2.  **Faker Script:** This function generates a small batch of synthetic e-commerce order data (as JSON) using `faker`. Crucially, it will intentionally inject data quality issues (nulls, bad formats, duplicates).
3.  **Bronze Layer (Ingestion):** The Cloud Function uploads this "dirty" JSON file to a GCS bucket.
4.  **Bronze -\> Silver (Cleansing):**
      * The GCS bucket is registered as a Dataplex **Bronze Zone** Asset.
      * We will have a BigQuery table `bronze.raw_orders` (configured as an external table over the GCS bucket) to read this raw data.
      * A BigQuery **Stored Procedure** (`sp_process_bronze_to_silver`) will run. This SP reads from the `bronze` table, applies all our data quality fixes (casting, handling nulls, filtering), and inserts the clean, structured data into the `silver.trusted_orders` table.
5.  **Silver -\> Gold (Aggregation):**
      * The `silver.trusted_orders` table is in a Dataplex **Silver Zone**.
      * A second Stored Procedure (`sp_process_silver_to_gold`) runs, reading from the clean `silver` table to create an aggregated "data product" in the `gold.daily_product_summary` table.
6.  **Dataplex Governance (The Showcase):**
      * **Data Quality:** We will set up a Dataplex **Data Quality Scan** on the `silver` and `gold` tables to monitor their health.
      * **Data Lineage:** Because we are using BigQuery to move data (`INSERT...SELECT...`), Dataplex will *automatically* discover and display the full lineage graph from `bronze` -\> `silver` -\> `gold`.

-----

### Step 1: Synthetic Data Generation (with DQ Issues)

This Python script uses `faker` to create order data. Notice the `random.choice` functions intentionally introducing "bad" data.

**`main.py` (for a Cloud Function):**

```python
import functions_framework
import json
import random
import uuid
from datetime import datetime
from faker import Faker
from google.cloud import storage

# --- Configuration ---
BUCKET_NAME = "your-bronze-gcs-bucket-name"  # 1. TODO: Change this
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)
fake = Faker()
# ---------------------

def generate_bad_data():
    """Generates a single order record with intentional DQ issues."""
    
    # Issue 1: NULL customer_id
    customer_id = fake.uuid4()
    if random.random() < 0.1:  # 10% chance of being null
        customer_id = None
        
    # Issue 2: Badly formatted product_id (should be 'PROD-')
    product_id = f"PROD-{random.randint(1000, 9999)}"
    if random.random() < 0.15: # 15% chance of bad format
        product_id = f"sku_{random.randint(1000, 9999)}"

    # Issue 3: Quantity as string or negative
    quantity = random.randint(1, 5)
    if random.random() < 0.1:
        quantity = str(quantity) # String instead of int
    elif random.random() < 0.05:
        quantity = -1 # Negative value

    # Issue 4: Price as 0 or NULL
    price = round(random.uniform(5.50, 99.99), 2)
    if random.random() < 0.1:
        price = None
    elif random.random() < 0.05:
        price = 0.0

    # Issue 5: Duplicate order_id (we'll simulate this by sometimes re-using a static one)
    order_id = str(uuid.uuid4())
    if random.random() < 0.05:
        order_id = "STATIC_DUPLICATE_ID"

    return {
        "order_id": order_id,
        "customer_id": customer_id,
        "product_id": product_id,
        "quantity": quantity,
        "price": price,
        "order_timestamp": fake.date_time_this_month().isoformat()
    }

@functions_framework.http
def generate_synthetic_data(request):
    """
    HTTP-triggered Cloud Function to generate and upload synthetic data.
    Trigger this with Cloud Scheduler.
    """
    data_batch = [generate_bad_data() for _ in range(10)] # Generate 10 orders per file
    
    # Convert to JSON Lines format (one JSON object per line)
    jsonl_data = "\n".join(json.dumps(record) for record in data_batch)
    
    # Define file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"raw_orders/orders_{timestamp}.jsonl"
    
    # Upload to GCS
    blob = bucket.blob(file_name)
    blob.upload_from_string(jsonl_data, content_type="application/jsonl")
    
    print(f"Successfully uploaded {file_name} to {BUCKET_NAME}.")
    return f"OK: Uploaded {file_name}", 200

```

**`requirements.txt`:**

```
functions-framework
google-cloud-storage
Faker
```

-----

### Step 2: BigQuery Medallion Setup (SQL)

In the BigQuery console, run this SQL to create your datasets and tables.

```sql
-- 1. Create the Datasets (Bronze, Silver, Gold)
CREATE SCHEMA IF NOT EXISTS bronze_layer;
CREATE SCHEMA IF NOT EXISTS silver_layer;
CREATE SCHEMA IF NOT EXISTS gold_layer;

-- 2. Create the Bronze Table (External Table over GCS)
-- This table maps directly to your GCS bucket.
-- NOTE: You MUST create the table from the BQ UI by selecting
-- "Create Table" -> "Create table from: Google Cloud Storage"
-- Set Table type: External table
-- Set "Path from GCS": your-bronze-gcs-bucket-name/raw_orders/*
-- Set File format: JSONL
-- Set Table name: bronze_layer.raw_orders
-- Check "Autodetect" for schema, or define it as all STRINGs initially.
-- For this demo, let's assume we define it with the final *types*
-- but BQ will read them as strings from JSONL anyway.

-- After creating the external table 'bronze_layer.raw_orders' manually,
-- you can proceed with the Silver and Gold tables.

-- 3. Create the Silver Table (Clean, Trusted Data)
CREATE TABLE IF NOT EXISTS silver_layer.trusted_orders (
  order_id        STRING NOT NULL OPTIONS(description="Unique order identifier"),
  customer_id     STRING NOT NULL OPTIONS(description="Unique customer identifier"),
  product_id      STRING NOT NULL OPTIONS(description="Unique product identifier (format PROD-NNNN)"),
  quantity        INT64  NOT NULL OPTIONS(description="Number of items, must be > 0"),
  price           FLOAT64 NOT NULL OPTIONS(description="Price per item, must be > 0"),
  total_price     FLOAT64 NOT NULL OPTIONS(description="Calculated field: quantity * price"),
  order_timestamp TIMESTAMP NOT NULL OPTIONS(description="Time the order was placed"),
  processing_timestamp TIMESTAMP NOT NULL OPTIONS(description="When this record was processed into Silver")
);

-- 4. Create the Gold Table (Aggregated Data Product)
CREATE TABLE IF NOT EXISTS gold_layer.daily_product_summary (
  product_id    STRING NOT NULL,
  sale_date     DATE NOT NULL,
  total_units_sold INT64,
  total_revenue    FLOAT64,
  processing_timestamp TIMESTAMP NOT NULL
);
```

-----

### Step 3: Stored Procedures (The Pipeline Logic)

These are the engines of your pipeline. They contain the data quality and business logic.

```sql
-- 1. Stored Procedure: Bronze -> Silver (Cleansing & Standardization)
CREATE OR REPLACE PROCEDURE silver_layer.sp_process_bronze_to_silver()
BEGIN
  -- Use MERGE to handle idempotency and avoid duplicates
  MERGE silver_layer.trusted_orders AS T
  USING (
    -- This subquery is the core DQ and transformation logic
    WITH cleaned_data AS (
      SELECT
        order_id,
        customer_id,
        product_id,
        -- DQ Fix: Handle quantity as string or int
        SAFE_CAST(quantity AS INT64) AS quantity, 
        -- DQ Fix: Handle price as string or float
        SAFE_CAST(price AS FLOAT64) AS price,
        -- DQ Fix: Parse timestamp
        SAFE_CAST(order_timestamp AS TIMESTAMP) AS order_timestamp
      FROM
        bronze_layer.raw_orders
    )
    -- Apply business rules and filter out bad data
    SELECT
      order_id,
      customer_id,
      product_id,
      quantity,
      price,
      (quantity * price) AS total_price,
      order_timestamp,
      CURRENT_TIMESTAMP() AS processing_timestamp
    FROM
      cleaned_data
    WHERE
      order_id IS NOT NULL AND order_id != 'STATIC_DUPLICATE_ID'
      AND customer_id IS NOT NULL
      AND product_id IS NOT NULL AND STARTS_WITH(product_id, 'PROD-') -- DQ Rule: Enforce format
      AND quantity > 0  -- DQ Rule: Filter negative/zero quantities
      AND price > 0     -- DQ Rule: Filter negative/zero prices
      AND order_timestamp IS NOT NULL
  ) AS S
  ON T.order_id = S.order_id
  WHEN NOT MATCHED THEN
    INSERT (
      order_id, customer_id, product_id, quantity, price, ENTERPRISE,
      order_timestamp, processing_timestamp
    )
    VALUES (
      order_id, customer_id, product_id, quantity, price, total_price,
      order_timestamp, processing_timestamp
    );
END;


-- 2. Stored Procedure: Silver -> Gold (Aggregation)
CREATE OR REPLACE PROCEDURE gold_layer.sp_process_silver_to_gold()
BEGIN
  -- We'll overwrite the gold table each time for simplicity in a POC.
  -- A real-world scenario would use an incremental MERGE.
  CREATE OR REPLACE TABLE gold_layer.daily_product_summary AS
  SELECT
    product_id,
    DATE(order_timestamp) AS sale_date,
    SUM(quantity) AS total_units_sold,
    SUM(total_price) AS total_revenue,
    CURRENT_TIMESTAMP() AS processing_timestamp
  FROM
    silver_layer.trusted_orders
  GROUP BY
    1, 2;
END;
```

-----

### Step 4: Pipeline Orchestration (Serverless)

We will use Cloud Functions and Scheduler to make this event-driven and automated.

1.  **Data Generator Function:**

      * Deploy the Python script from **Step 1** as a Cloud Function.
      * Set the trigger to **HTTP**. Note the trigger URL.

2.  **Scheduler Job:**

      * Go to **Cloud Scheduler**.
      * Create a job.
      * **Frequency:** `*/2 * * * *` (This is cron syntax for "every 2 minutes").
      * **Target:** `HTTP`.
      * **URL:** Paste the trigger URL from your Cloud Function.
      * **HTTP Method:** `POST` (or `GET`, depending on your CF setup).

3.  **Data Processing Function:**

      * This is the *second* Cloud Function.
      * **Trigger:** Set to **Cloud Storage**, Event type: `google.storage.object.finalize`, and point it at your `your-bronze-gcs-bucket-name`.
      * This function's `main.py` will be very simple. It just needs to call the Stored Procedures.

**`main.py` (for the *processing* Cloud Function):**

```python
import functions_framework
from google.cloud import bigquery

client = bigquery.Client()

@functions_framework.cloud_event
def process_data(cloud_event):
    """
    Triggered by GCS file upload. Calls the BQ Stored Procedures.
    """
    data = cloud_event.data
    bucket = data["bucket"]
    file_name = data["name"]

    print(f"Processing file: {file_name} from bucket: {bucket}.")

    try:
        # 1. Call the Bronze -> Silver SP
        print("Calling Bronze-to-Silver SP...")
        client.query("CALL silver_layer.sp_process_bronze_to_silver();").result()
        print("Bronze-to-Silver complete.")

        # 2. Call the Silver -> Gold SP
        print("Calling Silver-to-Gold SP...")
        client.query("CALL gold_layer.sp_process_silver_to_gold();").result()
        print("Silver-to-Gold complete.")
        
        return "OK", 200

    except Exception as e:
        print(f"Error processing pipeline: {e}")
        return "Error", 500
```

**`requirements.txt`:**

```
functions-framework
google-cloud-bigquery
```

-----

### Step 5: The Dataplex Showcase (Your Key Goal)

This is where you demonstrate the business value.

#### 1\. Setup Your Dataplex Lake

1.  Go to the **Dataplex** service in the GCP Console.
2.  Click **Create Lake**. Give it a name (e.g., `company-data-lake`).
3.  Inside your Lake, create three **Zones**:
      * **Zone 1:**
          * Name: `bronze-zone`
          * Type: **Raw**
          * Asset: Click "Add Asset". Select **GCS Bucket** and point it to `your-bronze-gcs-bucket-name`.
      * **Zone 2:**
          * Name: `silver-zone`
          * Type: **Curated**
          * Asset: Click "Add Asset". Select **BigQuery Dataset** and point it to `silver_layer`.
      * **Zone 3:**
          * Name: `gold-zone`
          * Type: **Curated**
          * Asset: Click "Add Asset". Select **BigQuery Dataset** and point it to `gold_layer`.

Let Dataplex discover the assets. This can take a few minutes.

#### 2\. Showcase: Data Quality (DQ)

1.  In Dataplex, navigate to the **Manage Quality** tab.
2.  Click **Create Data Quality Scan**.
3.  **Select Data:** Point it to your `silver_layer.trusted_orders` table.
4.  **Configure Rules:** This is the magic.
      * Click **Add Rule**.
      * **Rule 1 (Built-in):** Select `Row condition`. Rule type: `Check for nulls`. Column: `customer_id`.
      * **Rule 2 (Built-in):** Select `Row condition`. Rule type: `Check for nulls`. Column: `order_id`.
      * **Rule 3 (Custom SQL):** Select `Custom SQL statement`.
          * `Statement`: `SELECT COUNT(*) FROM table WHERE quantity <= 0`
          * `Threshold`: Set to `0` (meaning, fail if any record has `quantity <= 0`).
      * **Rule 4 (Custom SQL):** Select `Custom SQL statement`.
          * `Statement`: `SELECT COUNT(*) FROM table WHERE price <= 0`
          * `Threshold`: Set to `0`.
      * **Rule 5 (Custom SQL):** Select `Custom SQL statement`.
          * `Statement`: `SELECT COUNT(*) FROM table WHERE NOT STARTS_WITH(product_id, 'PROD-')`
          * `Threshold`: Set to `0`.
5.  **Schedule:** Set it to run on a schedule (e.g., hourly or daily).
6.  **Run** the scan.
7.  **Value:** After it runs, you will get a dashboard showing "Pass" or "Fail" for each rule. You can show how, *even though our Silver SP tries to clean data*, this scan acts as an *independent auditor* to verify the quality. If a new, bad `product_id` format appears, this scan will catch it.

#### 3\. Showcase: Data Lineage

This is the easiest and most impressive part. **You don't have to do anything.**

1.  Let your pipeline run for a few cycles (wait \~10 minutes so a few files are processed).

2.  In Dataplex, go to the **Search** tab.

3.  Search for your gold table: `daily_product_summary`.

4.  Click on the search result.

5.  On the table's details page, click the **Lineage** tab.

6.  **Value:** You will see a visual, automatically-generated graph that looks like this:

    `[bronze_layer.raw_orders]` -\> `[silver_layer.trusted_orders]` -\> `[gold_layer.daily_product_summary]`

This graph is your showcase. You can explain that by using native BigQuery jobs (which our Stored Procedures are), Dataplex automatically tracks and visualizes the complete "farm-to-table" journey of the data, providing trust, auditability, and impact analysis.

### A Note on Dataflow and Composer

You mentioned Dataflow and Composer. For this POC, I used Cloud Functions and Scheduler because they are *significantly* easier to set up and are extremely friendly to the free tier.

  * **When to use Dataflow:** You would replace the `process-data` Cloud Function and the B-\>S Stored Procedure with a **Dataflow Job** if your transformations were extremely complex, required streaming (vs. micro-batch), or needed to join/process data from *outside* BigQuery (e.g., a Pub/Sub stream + a Cloud SQL table).
  * **When to use Cloud Composer:** You would replace the simple GCS-trigger-to-CF chain with a **Composer (Airflow) DAG** if you had complex dependencies. For example: "Only run the S-\>G job *after* the `trusted_orders` job AND the `trusted_customers` job have both successfully finished."

This serverless design gives you the same Medallion logic and Dataplex features in a more lightweight package perfect for a POC.

Would you like me to elaborate on how to configure the Dataflow or Composer alternatives?