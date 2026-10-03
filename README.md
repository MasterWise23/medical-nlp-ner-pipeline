# Medical NER Pipeline & FastAPI Service

A professional-grade Medical Named Entity Recognition (NER) pipeline built in Python, leveraging state-of-the-art Hugging Face Transformer models and exposing a high-performance web service via FastAPI.

## Architecture & Technologies
- **Language:** Python 3.10+
- **NLP / Machine Learning:** Hugging Face Transformers (\Helios9/BioMed_NER\), PyTorch
- **Web API:** FastAPI, Uvicorn, Pydantic
- **Post-processing:** Custom sub-word and adjacent entity merging algorithm based on character coordinates.

## Project Structure
- \pi.py\ - FastAPI web server and entity post-processing logic.
- \medical_ner.py\ - Core script for running inference via command line.
- \equirements.txt\ - Project dependencies.
- \.gitignore\ - Excluded files for version control.

## Installation & Local Run

1. **Clone the repository:**
   \\\ash
   git clone https://github.com/MasterWise23/medical-nlp-ner-pipeline.git
   cd medical-nlp-ner-pipeline
   \\\

2. **Create and activate virtual environment:**
   \\\ash
   python -m venv .venv
   .\\.venv\\Scripts\\Activate.ps1
   \\\

3. **Install dependencies:**
   \\\ash
   pip install -r requirements.txt
   \\\

4. **Start the API server:**
   \\\ash
   uvicorn api:app --reload
   \\\

5. **Testing:**
   Access the interactive Swagger UI documentation at: **http://127.0.0.1:8000/docs**
