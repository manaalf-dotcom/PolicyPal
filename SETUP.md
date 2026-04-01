# SETUP.md

## 1. Overview

This guide explains how to install and run **Gemini-PolicyPal** locally or deploy it online.

The system requires:

* Python 3.9+
* Required dependencies
* Google Gemini API key

---

## 2. Quick Setup (Automatic - Recommended for Windows)

For convenience, this project provides a one-click setup script.

### Step 1: Run setup script

```bash
setup.bat
```

This script will automatically:

* Create a virtual environment (`.venv`)
* Activate the environment
* Upgrade `pip`
* Install all dependencies from `requirements.txt`

---

### Step 2: Run the application

```bash
run.bat
```

Or manually:

```bash
streamlit run app.py
```

---

### Notes

* This script is **Windows only**
* Make sure Python (3.9+) is installed before running

---

## 3. Manual Setup (All Platforms)

### 3.1 Clone the Repository

```bash
git clone https://github.com/Saku000/Gemini-PolicyPal.git
cd Gemini-PolicyPal
```

---

### 3.2 Create Virtual Environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 3.3 Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Configure API Key

### 4.1 Create `.env` file

In the root directory:

```text
.env
```

---

### 4.2 Add Gemini API key

```text
GEMINI_API_KEY=your_api_key_here
```

---

## 5. Prepare Data

### 5.1 QA / Analysis Mode

Place policy PDFs into:

```text
data/qa_policies/
```

---

### 5.2 Comparison Mode

Place two policies into:

```text
data/compare_prod/policy_a/
data/compare_prod/policy_b/
```

---

## 6. Run the Application

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```


---

## 7. Summary

To set up the project:

1. Run `setup.bat` (recommended)
2. Add API key
3. Add PDF files
4. Run the app

The system should now be fully functional.
