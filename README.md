# Retrieval-Augmented Question Answering over UIDAI Documents

NLP course project (PS-5). This notebook compares answers generated without documents with answers grounded in official UIDAI material. It evaluates BM25, dense retrieval using `sentence-transformers/all-MiniLM-L6-v2`, and a hybrid retriever. `google/flan-t5-base` generates the answers. The question set contains 36 hand-written questions with expected answers and gold evidence phrases.

## Repository layout

```text
NLP_Project/
├── Assignment.ipynb                         # Run cells in order
├── Detailed Required Provided by Prof.txt  # Assignment brief
├── README.md
├── Documents/                              # Source PDFs
├── corpus_text/                            # Created by the notebook
├── results/                                # Created by the notebook
└── figures/                                # Created by the notebook
```

The notebook uses `Path.cwd()` as the project root. Start it with the working directory set to `NLP_Project`; `Documents/` must be directly beneath that directory. The `PDF_MAP` dictionary in `Assignment.ipynb` lists the 16 exact PDF filenames used. Other PDFs in `Documents/` are ignored.

### Source PDF still needed

The current `Documents/` directory is missing **`Standard operating Procedure (SOP) to be followed for updating name in Aadhaar.pdf`**. Download the document titled **“Standard operating Procedure (SOP) to be followed for updating name in Aadhaar”** from the [official UIDAI circulars page](https://uidai.gov.in/en/circulars?limit=10&page=0), place it in `Documents/`, and give it the exact filename shown above. The notebook stops with a list of missing PDFs until all 16 mapped files are present. `FAQ-Aadhar App.pdf` is already present and mapped to the app questions.

Keep a record of each PDF's source URL, access date, and version for the dataset card. [UIDAI's website policy](https://uidai.gov.in/en/website-policies) describes its reproduction and attribution conditions; credit UIDAI when sharing the documents or extracts.

## Run in Google Colab

1. Put `Assignment.ipynb` and `Documents/` together in a `NLP_Project` folder in Google Drive. Add the missing SOP PDF described above. You can alternatively clone the GitHub repository if it includes all source PDFs.
2. Open `Assignment.ipynb` in Colab. Before running its first cell, run this setup cell:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive/NLP_Project
   !apt-get update -qq
   !apt-get install -y -qq poppler-utils tesseract-ocr
   %pip install -q numpy pandas matplotlib torch transformers huggingface_hub sentencepiece
   ```

   Change the `%cd` path if your Drive folder has a different location. Confirm that `Documents/` is visible from that directory.
3. Restart the Colab runtime if the package installation requests it, then set the working directory again and run **Runtime → Run all**. Run the notebook cells in order. Internet access is needed for the first Hugging Face model download.
4. Inspect the missing-PDF check, gold-evidence validation, retrieval results, QA results, and error examples. Do not use the conclusion's fixed numbers in a report unless they match the new run.

The notebook specifies a fixed seed of **42**, uses CPU inference and greedy decoding, and does not fine-tune either model. Its header records Python 3.12 and the library versions used by the original author. Colab's versions can differ, so record the actual versions and model revisions from your final run. Budget **several hours on a Colab CPU** for OCR, embedding, and five QA runs; this is a planning estimate, not a measured benchmark. A GPU can speed up inference only after the notebook code is adapted to use it.

## Run locally

Use Python 3.12 if available. Install Poppler (`pdfinfo`, `pdftotext`, `pdftoppm`) and Tesseract (`tesseract`) so the commands are on `PATH`. From the `NLP_Project` directory, install the Python dependencies:

```bash
python -m pip install numpy pandas matplotlib torch transformers huggingface_hub sentencepiece
jupyter notebook Assignment.ipynb
```

On Debian or Ubuntu, the system packages are `poppler-utils` and `tesseract-ocr`. On Windows, install Poppler and Tesseract separately and add their executable directories to `PATH`.

## Execution order and outputs

Run every notebook cell from top to bottom:

1. Set paths and seed; extract text from the PDFs with OCR when needed.
2. Clean and chunk the corpus; validate the 36 gold evidence phrases.
3. Build BM25, dense, and hybrid indexes; calculate Recall@k and MRR.
4. Compare chunk sizes of 50, 100, and 200 words.
5. Generate direct, retrieval-grounded, and oracle answers; score and inspect them.

The notebook creates these outputs:

| Path | Contents |
|---|---|
| `corpus_text/` | One extracted text file per mapped PDF and `_manifest.json` |
| `results/retrieval_results.csv` | Recall@k, MRR, and retrieval timing |
| `results/ablation_chunk_size.csv` | Retrieval results by chunk size and method |
| `results/qa_results.csv` | Aggregate QA metrics by system |
| `results/qa_answers_per_question.csv` | Individual generated answers and scores |
| `figures/` | Retrieval, ablation, and QA plots |

If you replace or update a PDF, remove its old text in `corpus_text/` (or remove the entire `corpus_text/` folder) before rerunning extraction. The notebook skips extraction when text files for all mapped documents already exist.

## Scope and evaluation notes

The corpus consists of official, English-language UIDAI FAQs, enrolment forms, a resident handbook, an acceptable-documents list, and a name-update SOP. It is a narrow document collection, and its policies may change. The 36 questions are an evaluation set written from these documents; there is no model training split. The notebook reports Recall@k, MRR, answer correctness and completeness, evidence support, an unsupported-answer proxy, and a citation-faithfulness proxy. The last two rely on automatic lexical matching and require manual review for the final paper.

The assignment also requires a research paper, a dataset card, results and figures, a contribution statement from all four members, an AI-use declaration, and a submission archive. These are separate deliverables; this README and notebook do not by themselves complete the submission.
