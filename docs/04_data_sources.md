# 04 — Data Sources

> All data must be public, aggregate, and free of personal identifiers. No scraping — only data the publisher has explicitly released for download.

## Required data and where to get it

### A. Tourist arrivals (overall) — DSEC
- **Source**: DSEC (Statistics and Census Service, Macau SAR), Visitor Arrivals Statistics
- **Primary URL**: https://www.dsec.gov.mo/en-US/Statistic?id=402
- **Format**: XLSX (main tables) — preferred; quarterly PDF fast-release as backup
- **What we need**: Monthly total arrivals by transit point (Outer Harbour, Border Gate, Airport, etc.), 2019–2025
- **License**: Public, OK to redistribute statistics with attribution
- **Status**: ☐ acquired
- **Ingest command**: `python -m src.ingest_data --source dsec --input data/raw/dsec/arrivals_YYYYMMDD.xlsx`

**Step-by-step download (Excel — preferred):**
1. Open `https://www.dsec.gov.mo/en-US/Statistic?id=402` in a browser (requires JavaScript)
2. Scroll to **"Main tables"** → find **"Table 1: Visitor Arrivals by Month and Transit Point"**
3. Click **Download** → save as `data/raw/dsec/arrivals_YYYYMMDD.xlsx`
4. Alternative: `https://www.dsec.gov.mo/TourismDBWeb/#/main?lang=en` → Visitor Arrivals → Export Excel

**Backup: quarterly PDF fast-release (3 months each, download and place in `data/raw/dsec/quarterly_pdfs/`):**
```
Q2 2025: https://www.dsec.gov.mo/getAttachment/5f09bdbb-a5d3-4f05-9bf5-bafe45983d2b/E_TUR_FR_2025_Q2.aspx
Q4 2024: https://www.dsec.gov.mo/getAttachment/6dc2a848-1201-4d87-99ea-3ee4a18c90ba/E_TUR_FR_2024_Q4.aspx
Q4 2023: https://dsec.gov.mo/getAttachment/28f1ae0e-bd81-4915-baa8-ab68849a3555/E_TUR_FR_2023_Q4.aspx
Q3 2023: https://www.dsec.gov.mo/getAttachment/bdd1bdc6-c1d4-4dbc-adce-e22e95353f24/E_TUR_FR_2023_Q3.aspx
Q4 2019: https://www.dsec.gov.mo/getAttachment/00eee72e-8e7a-4a7a-baec-541d2b81a8c8/E_TUR_FR_2019_Q4.aspx
```
(Parse with `src.utils.parse_dsec.parse_quarterly_pdf()` — requires `pip install pdfplumber`)

### B. Per-attraction visitor counts — MGTO
- **Source**: Macao Government Tourism Office (MGTO), Annual Reports / Macao Yearbook Tourism chapter
- **Primary URL**: https://www.macaotourism.gov.mo/en/about-mgto/publications/annual-report
- **Backup URL**: https://yearbook.gcs.gov.mo → Tourism chapter PDFs
  - e.g. 2024 edition: `https://yearbook.gcs.gov.mo/yearbook_pdf/2025/myb2025ePA01CH14.pdf`
- **Format**: PDF reports — per-attraction counts NOT available as Excel
- **What we need**: Annual visitor counts at named attractions (Ruins of St. Paul's, A-Ma Temple, etc.), 2019–2024
- **License**: Public reports, OK with attribution
- **Status**: ☐ acquired
- **Ingest command**: `python -m src.ingest_data --source mgto --fallback`

**Step-by-step data entry:**
1. Open MGTO Annual Reports (link above) for years 2019–2024
2. Find table: **"Number of Visitors to Major Tourist Attractions"**
3. Fill `data/raw/mgto/attractions_manual.csv` (template pre-populated with all node_id × year rows)
4. Run: `python -m src.ingest_data --source mgto`

**Synthetic fallback** (if MGTO PDF data is unavailable):
- `python -m src.ingest_data --source mgto --fallback`
- Uses `annual_visitors_est` from `src/utils/attractions.py` as proportional proxy
- confidence="estimate"; report must note: "Per-attraction proportions from estimated baselines"

**Note**: MGTO DataPlus (`https://dataplus.macaotourism.gov.mo/Publication/Other?lang=E`) has
Excel files for aggregate monthly arrivals (total to Macau) but NOT per-attraction heritage site counts.
Those are only in the PDF Annual Reports.

### C. Spatial / network topology
- **Source**: OpenStreetMap
- **Tool**: `osmnx` Python library
- **What we need**: Walking-network graph of Macau historic centre (~1.5 km × 1.5 km box), node coordinates, edge lengths
- **License**: ODbL — full redistribution permitted with attribution
- **Acquisition**:
  ```python
  import osmnx as ox
  bbox = (22.205, 22.190, 113.545, 113.530)  # rough Macau historic centre
  G = ox.graph_from_bbox(*bbox, network_type='walk')
  ```
- **Status**: ☐ acquired

### D. Calendar data (holidays, events, weather)
- **Source**: SMG (geophysical and meteorological bureau) for weather; public holiday calendar for events
- **URL**: https://www.smg.gov.mo
- **What we need**: Daily mean temperature, precipitation, Golden Week / public holiday flags
- **Use**: Covariates in arrival rate model
- **Status**: ☐ acquired

### E. Ground-truth crowd density (validation only, optional)
- **Source**: Self-collected via timed visits to bottleneck locations
- **Method**: Counting persons in fixed area at fixed intervals (e.g. every 30 min for 4 hours per location, 2–3 locations)
- **What we need**: 2–3 days of fieldwork data for cross-validation
- **Ethics**: No filming, no personal info — only count. Add note in report.
- **Status**: ☐ collected (target: one weekend in July)

## Data schemas

### `data/processed/arrivals_monthly.parquet`
| column | type | description |
|---|---|---|
| year_month | datetime | first day of month |
| origin | categorical | mainland / hk / taiwan / intl_asia / intl_other |
| transit_point | categorical | ferry_outer / ferry_inner / border_gate / airport |
| count | int64 | number of arrivals |

### `data/processed/attractions.parquet`
| column | type | description |
|---|---|---|
| node_id | str | matches graph node id |
| name_en | str | English name |
| name_zh | str | Chinese name |
| lat | float | latitude |
| lon | float | longitude |
| annual_visitors_est | int | best estimate of annual visitors |
| confidence | str | "direct" / "estimated" / "modeled" |

### `data/processed/graph.gml` (or `.graphml`)
- Networkx-compatible graph with walking edges
- Node attributes: `name_zh`, `name_en`, `lat`, `lon`, `type` (attraction/transit/junction)
- Edge attributes: `length_m`, `walk_time_s`

## Data licensing notes
- DSEC publications: Public, cite "Source: DSEC, Macau SAR" on every figure
- MGTO publications: Public, cite "Source: MGTO Yearbook YYYY"
- OSM: "© OpenStreetMap contributors" required on any map figure
- Self-collected fieldwork: Cite "Authors' field measurements, July 2026"

## Acquisition workflow
1. Check `data/raw/MANIFEST.txt` for what we already have
2. Download new data into `data/raw/<source>/<YYYYMMDD>_<description>.<ext>`
3. Write a parser in `src/utils/data_<source>.py`
4. Run parser, save to `data/processed/`
5. Update `MANIFEST.txt` with date, source URL, file hash
6. Record the acquisition in `MANIFEST.txt`

## What we will NOT use
- ❌ Non-heritage venue data (out of scope for heritage-corridor congestion)
- ❌ Social media scraped data — terms of service unclear
- ❌ Any data behind a login
- ❌ Any data identifying individuals (CCTV, mobile signals, etc.)
- ❌ Data without clear public-domain or open license
