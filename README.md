# UK Housing Price Analysis
 
Unlock the UK housing market’s hidden patterns — compare areas, track price trends, and spot the best opportunities for buying, selling, or investing.

---

## 📂 Getting Started

### 1. Download UK Housing Data

This dashboard **does not ship with full data**. You must manually download UK Price Paid Data from the official UK Government portal:

📎 [Price Paid Data – GOV.UK](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads)

* Choose **CSV format**
* You can select individual months or the full historical archive [Note: App will not support full dataset.]
* This tool supports any region, but the default demo shows Gloucestershire.

Once downloaded, upload the CSV via the **"Upload dataset"** section in the Streamlit sidebar.

---

### 2. Run the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔍 What You Can Do With This Dashboard

This app is built for users interested in:

### 🏠 **Buying a Home**

* See historical **price trends** in your target postcode or town
* Understand seasonal patterns and local pricing mix
* Compare **Freehold vs Leasehold** and **New vs Old builds**

### 📊 **Selling or Valuing a Property**

* Track **recent transactions** in your street
* Benchmark your home’s price vs nearby areas or similar property types
* Spot **outliers or unusual transactions** near your address

### 📈 **Property Investment**

* Use **area rankings** to find places with high median growth
* Analyze **repeat-sale returns** to estimate typical resale performance
* Run simple **hedonic models** to isolate the effect of size, tenure, and geography

---

## 💡 Tips for Using the Dashboard

* Use **date filters** to focus on recent or historical windows
* Enable **winsorisation** to remove the impact of extreme values
* Read the **caption under each section** – they help explain how visuals work
* Export charts and tables from the **three-dot menu (⋮)** in any block

---

## Data Access URL
```angular2html
http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-{year}.csv
```