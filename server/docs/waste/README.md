Drop waste-sorting / recycling guideline PDFs, DOCX, or HTML files here.

On first run with no files here, `waste_vector.py` falls back to a small
built-in seed document so the feature still works. Once you add real files,
delete `server/chroma_db_waste/` so it rebuilds the vector store from them.
