# UK House Address Parser

UK address strings in EPC data are messy and inconsistent, which makes linking to cleaner housing datasets hard.  
This project builds a trainable parser that turns free-text addresses into a structured schema (PAON, SAON, Street, Locality, Post Town, Postcode, …).  
A curated **address base** (exported as `.parquet`) is used for training, validation, and lookups.

---

## Features

- **Token-level parser** for UK addresses (supports ranges and suffixes like `12–14`, `12A`).
- **UK-aware normalisation**: synonyms, street types, post town and postcode checks.
- **Pluggable models**: CRF baseline and a modern NER option (e.g., BERT-CRF).
- **Post-processing** to extract PAO/SAO numbers/suffixes and fix postcode spacing.

---

## Data

* **Training base**: exported housing dataset in `.parquet` with canonical fields.
* **Helper lookups**: `data/` holds counties, localities, synonyms, outcodes↔post towns, etc.
* **Licensing**: if your source comes from Royal Mail PAF or OS AddressBase, make sure your licence allows model training and derived works.

---

## Project layout

```
.
├─ data/
│  ├─ raw/            # EPC dumps, etc. (not committed)
│  ├─ interim/        # cleaned/normalised
│  ├─ processed/      # training-ready
│  └─ lookups/        # counties.csv, synonyms.csv, postcode_district_to_town.csv
├─ src/address_parser/
│  ├─ models/         # crf.py, transformer.py
│  ├─ rules/          # postcode validators, PAO/SAO extractors
│  ├─ io/             # readers/writers
│  ├─ train.py        # training entrypoint
│  ├─ infer.py        # inference entrypoint
│  └─ cli.py          # CLI wrapper
├─ configs/           # YAML configs for data/model/train
├─ experiments/       # run logs, metrics, cfg snapshots
├─ tests/             # unit/integration tests
└─ reports/           # parsed outputs, figures, tables
```

---

## Models

* **CRF baseline**
  Linear-chain CRF with hand-crafted features (digits, street types, post towns, outcodes, etc.).
  Fast and easy to debug.

* **NER (Transformer + CRF head)**
  Treats fields as entities (`SAON`, `PAON`, `STREET`, `LOCALITY`, `POST_TOWN`, `POSTCODE`, …) with BIO tags.
  Trains on rendered variants from the address base. Usually higher recall on messy inputs.

You can keep the postcode/post-town checks and PAO/SAO regex logic for both models.


## Roadmap

* [ ] Parser method to emit **PAON/SAON/etc. as XML** (for ONS-style pipelines)
* [ ] Train and compare **CRF vs BERT-CRF** on held-out postcode areas
* [ ] Refresh helper files (counties, localities, synonyms) from latest datasets
* [ ] Add optional **record linkage** step using [`recordlinkage`](https://github.com/J535D165/recordlinkage) for partial matches
* [ ] CLI/API for batch parsing with confidence scores
* [ ] Dockerfile + minimal FastAPI service

---

## Contributing

Issues and PRs welcome. Please include:

* a minimal failing example,
* expected vs actual output,
* dataset sample (sanitised).

Run checks before pushing:

```bash
make lint test
```

---

## Credits

This project draws inspiration from ONSdigital’s Address Index work (parser structure and CRF approach).
Their repository does not include trained models or datasets.

* ONSdigital Address Index (reference code):
  [https://github.com/ONSdigital/address-index-data/blob/develop/DataScience/ProbabilisticParser/parser.py](https://github.com/ONSdigital/address-index-data/blob/develop/DataScience/ProbabilisticParser/parser.py)
