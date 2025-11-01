### Dataplex Universal Catalog ### 
is an intelligent data fabric that enables organizations to centrally discover, manage, monitor, and govern their data across data lakes, data warehouses, and data marts to power analytics at scale. 
Specifically, you can use Dataplex Universal Catalog to build a data mesh architecture, which is an organizational and technical approach that decentralizes data ownership among domain data owners.

Dataplex Universal Catalog manages data in a way that doesn’t require data movement or duplication. As you add new data assets, Dataplex Universal Catalog harvests the metadata for both structured and unstructured data, 
and automatically registers all metadata in a secure, unified metastore. Data and metadata can then be assessed via Google Cloud services such as BigQuery.

### Create a lake
In Dataplex Universal Catalog, a lake is the highest organizational domain that represents a specific data area or business unit. For example, you can create a lake for each department or data domain in your organization, so that you can organize and provide data for specific user groups.

In this task, you create a lake to start building a data mesh.

In the Google Cloud Console, select Navigation menu (Navigation menu) > View all products. Under Analytics, select Dataplex Universal Catalog.
If prompted Welcome to the new Dataplex Universal Catalog experience, click Close.

Under Manage lakes, click Manage.

Click +Create lake.

### Add a zone to your lake
After you create a lake, you can add zones to the lake. Zones are subdomains within a lake that you can use to categorize data further. For example, you can categorize data by stage, usage, or restrictions.

There are two types of zones:

Raw zones contain data in raw formats (such as files in Cloud Storage buckets) and are not subject to strict type-checking.
Curated zones contain data that is cleaned, formatted, and ready for analytics such as BigQuery tables.

### Attach an asset to a zone
Data stored in Cloud Storage buckets or BigQuery datasets can be attached as assets to zones within a Dataplex Universal Catalog lake.

In this task, you attach a Cloud Storage bucket that you create in the Google Cloud console.

On the Zones tab, click on the name of your zone.

On the Assets tab, click +ADD ASSET.

Click +ADD AN ASSET.
