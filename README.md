# 📄 Chat With Your PDF

Upload course notes or a handbook and ask questions. Answers include page references
from the uploaded document, plus expandable source passages so you can check them.

## Features
- Upload a text-based PDF (up to 30 MB and 500 pages).
- Ask questions in a chat. Follow-up questions work too.
- Answers cite pages like *(Page 4)* and show the passages they came from.
- Works without AI too: with no Groq key it shows the most relevant passages.
- Download the chat as a text file. Nothing is stored permanently.

## How it works
1. `pypdf` reads the text of each page.
2. The text is cut into small sections, each remembering its page number.
3. A built-in keyword search (BM25) finds the sections that match your question.
4. Those sections are sent to Groq, which writes an answer using only that text and cites the pages.

## Files
| File | Purpose |
|---|---|
| `app.py` | The whole Streamlit app |
| `requirements.txt` | Python packages (streamlit, pypdf, groq) |
| `.gitignore` | Keeps secrets and PDFs out of GitHub |

## Groq API key (recommended)
1. Create a free key at https://console.groq.com (API Keys → Create API Key).
2. On Streamlit Community Cloud: app **Settings → Secrets**, paste one line with straight quotes:
   ```
   GROQ_API_KEY = "gsk_your_key_here"
   ```
3. Save and **Reboot app**.

Locally, put the same line in `.streamlit/secrets.toml` (already in `.gitignore`). Never put the key in `app.py`.

## Run on your computer (optional)
```
pip install -r requirements.txt
streamlit run app.py
```

## Limits
- Scanned PDFs (images of pages) have no text and will not work without OCR.
- Page numbers follow the PDF's page order, which can differ from printed page numbers.
- Search is keyword-based, so use words that appear in the document.

## Storage note
No database is needed. To remember documents or chats later, use a free hosted database such as Neon PostgreSQL;
Streamlit Community Cloud erases local files (including SQLite) when the app restarts.
