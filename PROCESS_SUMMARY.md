# GenAI Analysis Process Summary
**Student ID:** rocky07 | **Assignment:** InfiGen GenAI Analyst

---

## Data Processing Pipeline

### 1. Data Sources
- **Media Articles**: 50 research articles (Excel)
- **Twitter Posts**: 50 healthcare professional tweets (Excel)

### 2. Data Combination Process
```
Articles: Title + Content → Combined Text
Twitter:  Handle + Posts → Combined Text
Result:   100 unified records with "Combined" column
```

### 3. Source Labeling
- Media articles tagged as "Media"
- Twitter posts tagged as "Twitter"

---

## NLP Models & Tasks

| Model | Task | Purpose |
|-------|------|---------|
| **BioBERT** | Entity Recognition | Extract: Drugs, Diseases, Study Names |
| **BART** | Topic Classification | Categorize: Efficacy, PFS, OS, Safety, Opinion, Others |
| **RoBERTa** | Sentiment Analysis | Detect: Positive, Negative, Neutral sentiment |

### GPU Acceleration
- **Device**: MacBook Pro MPS (Metal Performance Shaders)
- **Performance**: 1.8 records/second processing speed
- **Optimization**: Half-precision (float16) for faster inference

---

## Results Summary

### Overall Statistics
- **Total Records**: 100 (50 articles + 50 tweets)
- **Processing Time**: 62.54 seconds
- **Processing Speed**: 1.8 records/second

### Sentiment Distribution
| Sentiment | Count | Percentage |
|-----------|-------|------------|
| **Neutral** | 65 | 65% |
| **Positive** | 30 | 30% |
| **Negative** | 5 | 5% |

### Topic Classification
| Topic | Count | Percentage |
|-------|-------|------------|
| **Others** | 49 | 49% |
| **General Opinion** | 22 | 22% |
| **Safety-Side Effects** | 12 | 12% |
| **Overall Survival (OS)** | 7 | 7% |
| **Efficacy-General** | 5 | 5% |
| **Progression Free Survival (PFS)** | 5 | 5% |

### Entity Extraction
- **Medical Entities Found**: 600+ total entities
- **Drugs Identified**: nivolumab, trastuzumab, metformin, etc.
- **Diseases Found**: breast cancer, lymphoma, diabetes, etc.
- **Studies Detected**: CheckMate-901, DESTINY-Breast02, etc.

### Source Comparison
| Metric | Media Articles | Twitter Posts |
|--------|----------------|---------------|
| **Positive Sentiment** | 8% | 46% |
| **Average Confidence** | 26.9% | 33.4% |
| **Entities per Record** | ~10 | ~2 |

---

## Technical Implementation

### Key Features
- GPU-accelerated processing (3-5x faster than CPU)
- Medical domain-specific models
- Assignment-specific topic categories
- Comprehensive error handling
- Real-time progress tracking

### Model Sources
- **BioBERT**: `alvaroalon2/biobert_diseases_ner`
- **BART**: `facebook/bart-large-mnli`
- **RoBERTa**: `cardiffnlp/twitter-roberta-base-sentiment-latest`

---

## Output Files

| File | Content | Size |
|------|---------|------|
| `predictions_rocky07.csv` | Complete analysis results | 410KB |
| `analysis_summary_rocky07.json` | Statistical summary | 473B |
| `genai_analyst_assignment_rocky07.py` | Full analysis script | - |

---

## Key Insights

### Content Analysis
- **Media articles** focus on clinical trials and research findings
- **Twitter posts** emphasize healthcare discussions and opinions
- **Medical terminology** highly prevalent across both sources

### Sentiment Patterns
- **Twitter more positive** (46%) vs articles (8%)
- **Research articles more neutral** (formal/scientific tone)
- **Low negativity overall** (5%) indicates optimistic healthcare discourse

### Topic Focus
- **Clinical research dominates** formal articles
- **General healthcare discussion** leads Twitter content
- **Survival outcomes** (OS/PFS) represent 12% of content

---

## Assignment Compliance

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Data Import & Exploration | Complete | Pandas analysis of Excel files |
| Data Preparation | Complete | Cleaning, renaming, standardization |
| Source Column Addition | Complete | Media/Twitter labeling |
| Data Consolidation | Complete | Vertical merge of datasets |
| Text Enhancement | Complete | Title + Body → Combined column |
| Entity Recognition | Complete | BioBERT for medical entities |
| Topic Classification | Complete | BART for assignment categories |
| Sentiment Analysis | Complete | RoBERTa for emotional tone |
| Output Generation | Complete | CSV export with predictions |

---

**Assignment Status: COMPLETED SUCCESSFULLY**  
**Ready for submission to InfiGen**
