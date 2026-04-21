# FactSphere Project

This repository contains the FactSphere hallucination-aware QA system built for the Agentic AI Applications Course.

## Files
1. `FactSphere.ipynb`: The Google Colab notebook containing the entire agentic pipeline and evaluation suite.
2. `app.py`: A Streamlit web application providing a UI for the same pipeline.
3. `.env.example`: A template for how to configure the Anthropic API key.

## Instructions to Run the Assessment Notebook
1. Open Google Colab (https://colab.research.google.com).
2. Upload `FactSphere.ipynb` to Colab.
3. In Colab, on the left sidebar, click the "key" icon to add a Secret named `ANTHROPIC_API_KEY` and paste your Anthropic API key in it. Ensure the toggle is switched to enable access. Alternatively, you can just set `os.environ["ANTHROPIC_API_KEY"] = "your-key"` in a notebook cell.
4. Run all cells sequentially. The first cell installs all dependencies. The last cell executes the evaluation of 10 designed queries.

## Instructions to Run the Streamlit UI Locally
1. Ensure you have Python 3.10+ installed.
2. Install the necessary packages locally:
   ```bash
   pip install langchain langgraph langchain-anthropic sentence-transformers chromadb wikipedia-api streamlit anthropic numpy scikit-learn pydantic
   ```
3. Set your API Key in the environment.
   - On Windows (PowerShell): `$env:ANTHROPIC_API_KEY="your-api-key"`
   - On Linux/Mac: `export ANTHROPIC_API_KEY="your-api-key"`
4. Start the application:
   ```bash
   streamlit run app.py
   ```
5. Your browser will automatically open to `http://localhost:8501`.
