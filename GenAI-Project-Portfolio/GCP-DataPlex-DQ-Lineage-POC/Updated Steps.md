Here is a complete, end-to-end guide to building your POC.

This plan is designed to be **free-tier friendly** by using a serverless, event-driven architecture (Cloud Functions, GCS, BigQuery, Scheduler) instead of services like Composer or Dataflow which can incur costs and are more complex for a 2-minute batch.

This guide is structured in the precise order of execution to prevent any errors.

### 🚀 POC Architecture Overview

1.  **Generate:** A **Cloud Scheduler** job runs every 2 minutes.
2.  **Trigger:** It calls an **HTTP Cloud Function** (Python).
3.  **Create:** This function uses `faker` to generate "dirty" JSON data and dumps it into a **GCS (Bronze)** bucket.
4.  **Detect:** The GCS file drop triggers a *second* **Eventarc Cloud Function**.
5.  **Process:** This function executes two BigQuery **Stored Procedures**:
      * `sp_bronze_to_silver`: Cleans the raw data, fixes DQ issues, and loads it into the **Silver** BQ table.
      * `sp_silver_to_gold`: Aggregates the clean data into a **Gold** BQ data product.
6.  **Govern:** **Dataplex** is layered over all three assets to provide automated Data Quality rules and visualize the entire data lineage.

-----

### Step 1: GCP Project Setup & API Enablement

Before you start, do this once.

1.  Go to your GCP Console.
2.  Navigate to **"APIs & Services" \> "Library"**.
3.  Search for and **Enable** the following APIs:
      * Google Cloud Storage API
      * BigQuery API
      * Cloud Functions API
      * Cloud Scheduler API
      * Cloud Build API (used to deploy functions)
      * Eventarc API
      * Google Dataplex API
      * Cloud Resource Manager API (often needed by Dataplex)

-----

### Step 2: Storage Setup (GCS & BigQuery Datasets)

1.  **Create GCS Bucket (Bronze):**

      * Go to **Cloud Storage** \> **"Create Bucket"**.
      * Give it a **unique name** (e.g., `[your-project-id]-bronze-bucket`).
      * Choose a **Region** (e.g., `us-central1`). **Remember this region.**
      * Click **Create**.
      * Inside the new bucket, create a folder named `raw_orders`.

2.  **Create BigQuery Datasets (Medallion):**

      * Go to **BigQuery**.
      * In the Explorer panel, click the three dots (⋮) next to your project ID and select **"Create dataset"**.
      * **Dataset 1:**
          * Dataset ID: `bronze_layer`
          * Location: **Use the same region as your GCS bucket** (e.g., `us-central1`).
      * **Dataset 2:**
          * Dataset ID: `silver_layer`
          * Location: Same as above.
      * **Dataset 3:**
          * Dataset ID: `gold_layer`
          * Location: Same as above.

-----

### Step 3: Data Generation (Cloud Function \#1)

This function uses `faker` to create our "dirty" data.

1.  Go to **Cloud Functions** \> **"Create Function"**.
2.  Set **Environment** to `1st gen`.
3.  **Function Name:** `generate-synthetic-data`
4.  **Region:** The same region you've been using.
5.  **Trigger:**
      * Trigger type: `HTTP`
      * Authentication: `Allow unauthenticated invocations` (simpler for this POC).
      * Click **"Save"**.
6.  Click **"Next"**.
7.  On the Code page:
      * **Runtime:** `Python 3.10` (or newer)
      * **Entry point:** `generate_synthetic_data` (this is the Python function name)
      * Select the `main.py` tab and paste this code. **Remember to change `BUCKET_NAME`\!**

**`main.py`**

```python
import functions_framework
import json
import random
import uuid
from datetime import datetime
from faker import Faker
from google.cloud import storage

# --- !! CHANGE THIS !! ---
BUCKET_NAME = "your-unique-bronze-bucket-name"  # e.g., "[your-project-id]-bronze-bucket"
# -------------------------

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)
fake = Faker()

def generate_bad_data():
    """Generates a single order record with intentional DQ issues."""
    
    # Issue 1: NULL customer_id
    customer_id = fake.uuid4()
    if random.random() < 0.1:  # 10% chance
        customer_id = None
        
    # Issue 2: Badly formatted product_id
    product_id = f"PROD-{random.randint(1000, 9999)}"
    if random.random() < 0.15:
        product_id = f"sku_{random.randint(1000, 9999)}"

    # Issue 3: Quantity as string or negative
    quantity = random.randint(1, 5)
    if random.random() < 0.1:
        quantity = str(quantity)
    elif random.random() < 0.05:
        quantity = -1 

    # Issue 4: Price as 0 or NULL
    price = round(random.uniform(5.50, 99.99), 2)
    if random.random() < 0.1:
        price = None
    elif random.random() < 0.05:
        price = 0.0

    # Issue 5: Duplicate order_id
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
    """ HTTP-triggered Cloud Function to generate and upload synthetic data. """
    data_batch = [generate_bad_data() for _ in range(10)] # 10 orders per batch
    
    # Data is in JSON Lines format (one JSON object per line)
    jsonl_data = "\n".join(json.dumps(record) for record in data_batch)
    
    # Define file name in the folder we created
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"raw_orders/orders_{timestamp}.jsonl"
    
    # Upload to GCS
    blob = bucket.blob(file_name)
    blob.upload_from_string(jsonl_data, content_type="application/jsonl")
    
    print(f"Successfully uploaded {file_name} to {BUCKET_NAME}.")
    return f"OK: Uploaded {file_name}", 200
```

  * Select the `requirements.txt` tab and paste this:

**`requirements.txt`**

```
google-cloud-storage
Faker
functions-framework
```

8.  Click **"Deploy"**. This may take a few minutes.
9.  After it deploys, go to the **"Trigger"** tab and copy the **Trigger URL**. You'll need it now.

-----

### Step 4: Data Scheduling (Every 2 Mins)

1.  Go to **Cloud Scheduler** \> **"Create Job"**.
2.  **Name:** `run-data-generator`
3.  **Region:** Same region.
4.  **Frequency:** `*/2 * * * *` (This is cron syntax for "every 2 minutes")
5.  **Timezone:** Your choice (e.g., `America/New_York`)
6.  **Target type:** `HTTP`
7.  **URL:** Paste the **Trigger URL** from your Cloud Function in Step 3.
8.  **HTTP method:** `GET`
9.  Click **"Create"**.
10. **Test it:** Click **"Run Now"** on the job. Go to your GCS bucket and check the `raw_orders` folder. A new JSONL file should appear.

-----

### Step 5: BigQuery Table Creation

This is the **critical step** that caused your previous error. We must do this *before* creating the Stored Procedures.

#### 5a. Create Bronze External Table (Manual UI)

1.  In the BigQuery UI, find your `bronze_layer` dataset.
2.  Click the three dots (⋮) and select **"Create table"**.
3.  **Create table from:** `Google Cloud Storage`
4.  **Select file from GCS bucket:** `[your-unique-bronze-bucket-name]/raw_orders/*` (Use your bucket name and add `/*` at the end)
5.  **File format:** `JSONL (Newline delimited JSON)`
6.  **Dataset:** `bronze_layer`
7.  **Table:** `raw_orders`
8.  **Table type:** `External table`
9.  **Schema:** Check the **"Auto-detect"** box.
10. Click **"Create Table"**.
11. **Test it:** Run `SELECT * FROM bronze_layer.raw_orders LIMIT 10`. You should see your "dirty" data.

#### 5b. Create Silver & Gold Managed Tables (SQL)

Now run this SQL in the BigQuery editor to create the empty Silver and Gold tables.

```sql
-- Create the Silver Table (Clean, Trusted Data)
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

-- Create the Gold Table (Aggregated Data Product)
CREATE TABLE IF NOT EXISTS gold_layer.daily_product_summary (
  product_id    STRING NOT NULL,
  sale_date     DATE NOT NULL,
  total_units_sold INT64,
  total_revenue    FLOAT64,
  processing_timestamp TIMESTAMP NOT NULL
);
```

-----

### Step 6: Pipeline Logic (Stored Procedures)

Now that all tables exist, you can safely create the Stored Procedures that reference them. Run this SQL in the BigQuery editor.

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
      order_id, customer_id, product_id, quantity, price, total_price,
      order_timestamp, processing_timestamp
    )
    VALUES (
      S.order_id, S.customer_id, S.product_id, S.quantity, S.price, S.total_price,
      S.order_timestamp, S.processing_timestamp
    );
END;

-- 2. Stored Procedure: Silver -> Gold (Aggregation)
CREATE OR REPLACE PROCEDURE gold_layer.sp_process_silver_to_gold()
BEGIN
  -- We overwrite the gold table each time for simplicity in a POC.
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

### Step 7: Pipeline Orchestration (Cloud Function \#2)

This function is the "glue" that runs the SPs when a new file arrives.

1.  Go to **Cloud Functions** \> **"Create Function"**.
2.  Set **Environment** to `2nd gen`. (2nd gen is better for Eventarc triggers).
3.  **Function Name:** `process-medallion-pipeline`
4.  **Region:** Same region.
5.  **Trigger:**
      * Trigger type: `Cloud Storage`
      * **Event Type:** `google.storage.object.finalize` (This means "on file creation")
      * **Bucket:** Browse and select your `[your-unique-bronze-bucket-name]`.
      * Click **"Save"**.
6.  Under **"Runtime, build..."**, expand the "Service Account" section and select `Compute Engine default service account` (for a POC, this is simplest and has BQ/Storage permissions).
7.  Click **"Next"**.
8.  On the Code page:
      * **Runtime:** `Python 3.10` (or newer)
      * **Entry point:** `process_data`
      * Paste this code into `main.py`:

**`main.py`**

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

    # --- IMPORTANT: Filter to only process files in the right folder! ---
    if not file_name.startswith("raw_orders/"):
        print(f"Skipping file: {file_name}. Not in 'raw_orders/' folder.")
        return "OK", 200

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

  * Select the `requirements.txt` tab and paste this:

**`requirements.txt`**

```
google-cloud-bigquery
functions-framework
cloudevents
```

9.  Click **"Deploy"**.

**Test the Pipeline:** Go back to your Cloud Scheduler job and click **"Run Now"**.

  * **Watch:** A file appears in GCS.
  * **Wait:** This triggers the `process-medallion-pipeline` function.
  * **Verify:** Go to BigQuery. Run `SELECT * FROM silver_layer.trusted_orders` and `SELECT * FROM gold_layer.daily_product_summary`. They should now have data\! Your pipeline is working.

-----

### Step 8: The Dataplex Showcase (The "Why")

Now for the payoff.

#### 8a. Setup Dataplex Lake & Zones

1.  Go to **Dataplex** in the GCP Console.
2.  Click **"Create Lake"**.
      * Name: `company-data-lake`
      * Region: Same as your BQ/GCS assets.
3.  Click **"Create"**.
4.  Inside your new lake, click **"Add Zone"**.
      * **Zone 1 (Bronze):**
          * Name: `bronze-zone`
          * Type: **Raw**
          * Click **"Add Asset"**. Select `Cloud Storage bucket` and point it to your `[your-unique-bronze-bucket-name]`.
      * **Zone 2 (Silver):**
          * Click **"Add Zone"** again.
          * Name: `silver-zone`
          * Type: **Curated**
          * Click **"Add Asset"**. Select `BigQuery dataset` and point it to `silver_layer`.
      * **Zone 3 (Gold):**
          * Click **"Add Zone"** again.
          * Name: `gold-zone`
          * Type: **Curated**
          * Click **"Add Asset"**. Select `BigQuery dataset` and point it to `gold_layer`.

Dataplex will now discover your assets. This may take a few minutes.

#### 8b. Showcase 1: Data Quality

This is how you prove your data is trustworthy *after* cleaning.

1.  In Dataplex, navigate to the **"Manage Quality"** tab.
2.  Click **"Create Data Quality Scan"**.
3.  **Data Source:** Browse and select your `silver_layer.trusted_orders` table.
4.  **Scan Type:** `Built-in (Recommended)`.
5.  **Configure Rules:**
      * **Rule 1 (Not Null):**
          * Rule type: `Check for nulls`
          * Column: `customer_id`
      * **Rule 2 (Not Null):**
          * Click **"Add Rule"**.
          * Rule type: `Check for nulls`
          * Column: `order_id`
      * **Rule 3 (Value Check):**
          * Click **"Add Rule"**.
          * Rule type: `Custom SQL statement`
          * Statement: `SELECT COUNT(*) FROM table WHERE quantity <= 0`
          * Threshold: `0` (Fail the job if any record has `quantity <= 0`)
      * **Rule 4 (Format Check):**
          * Click **"Add Rule"**.
          * Rule type: `Custom SQL statement`
          * Statement: `SELECT COUNT(*) FROM table WHERE NOT STARTS_WITH(product_id, 'PROD-')`
          * Threshold: `0`
6.  **Schedule:** Set it to `On-demand` for now.
7.  Click **"Create"**.
8.  Find your new scan in the list and click **"Run"**.
9.  **Showcase:** After it runs (a minute or two), click on the job. You will see a dashboard showing **Pass/Fail** for each rule. This is your "Data Quality" receipt. You can show how this independently audits the cleansing logic you built in the Stored Procedure.

#### 8c. Showcase 2: Data Lineage

This is the most impressive part and is **100% automatic**.

1.  Let your pipeline run a few times (wait 4-5 minutes).

2.  In Dataplex, go to the **"Search"** tab.

3.  Search for your gold table: `daily_product_summary`.

4.  Click the search result.

5.  On the table's details page, click the **"Lineage"** tab.

6.  **Showcase:** You will see a beautiful, automatically-generated graph. This graph will visually show:

    `[bronze_layer.raw_orders]` ➡️ `[silver_layer.trusted_orders]` ➡️ `[gold_layer.daily_product_summary]`

This graph proves you have end-to-end visibility. You can click on any node and see the tables and the `MERGE` / `CREATE TABLE AS` jobs that connect them. This is the core value proposition of Dataplex for governance.
