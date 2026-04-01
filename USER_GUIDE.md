# USER_GUIDE.md

## 1. Overview

This guide explains how to use **Gemini-PolicyPal**, an AI-powered system for analyzing and comparing insurance policy documents.

The application provides three main features:

1. Policy Analysis (Dashboard)
2. Question Answering (Ask Pal)
3. Policy Comparison (Compare)

---

## 2. Launching the Application

### Run locally

```bash
streamlit run app.py
```

Open in browser:

```text
http://localhost:8501
```

---

## 3. Application Interface

The app contains three main pages:

* **Dashboard**
* **Ask Pal**
* **Compare**

You can navigate using the sidebar.

---

## 4. Dashboard (Policy Analysis)

### Purpose

Analyze a single insurance policy and extract structured insights.

---

### Steps

1. Place a PDF file into:

```text
data/qa_policies/
```

2. Open the **Dashboard** page

3. Click **Analyze**

---

### Output

The system will display:

* Policy type
* Coverage details
* Deductible
* Exclusions
* Risk score
* Summary

---

### Example Use Cases

* Quickly understand a long insurance document
* Identify potential risks
* Extract key financial terms

---

## 5. Ask Pal (Question Answering)

### Purpose

Ask natural language questions about a policy using RAG.

---

### Steps

1. Ensure policy PDF is in:

```text
data/qa_policies/
```

2. Go to **Ask Pal**

3. Enter your question

Examples:

* "Is vandalism covered?"
* "What is the deductible?"
* "Are there exclusions?"

---

### Output

* AI-generated answer
* Based on retrieved document content

---

### Notes

* Answers are grounded in the document
* Performance depends on document quality

---

## 6. Compare (Policy Comparison)

### Purpose

Compare two insurance policies across multiple dimensions.

---

### Steps

1. Place two PDFs:

```text
data/compare_prod/policy_a/
data/compare_prod/policy_b/
```

2. Go to **Compare**

3. Click **Run Comparison**

4. You can also ask questions.

---

### Output

* Structured comparison
* Category scores:

  * Coverage
  * Cost
  * Risk
* Overall winner
* Trade-offs
* Radar chart visualization

---

### Example Use Cases

* Choosing between two insurance plans
* Evaluating cost vs coverage trade-offs

---

## 7. Best Practices

* Use **text-based PDFs** (not scanned images)
* Use clear and complete policy documents
* Ask specific questions for better results

---

## 8. Limitations

* Performance depends on document quality
* Some values may not be extracted perfectly
* Results may vary slightly due to LLM behavior

---

## 9. Troubleshooting

### No results displayed

* Check PDF is placed in correct folder
* Ensure PDF is readable

---

### Slow response

* Large documents may take longer to process

---

### API errors

* Ensure API key is correctly configured

---

## 10. Summary

Gemini-PolicyPal enables users to:

* Analyze complex policy documents
* Ask questions using AI
* Compare policies with structured insights

This makes insurance documents easier to understand and evaluate.
