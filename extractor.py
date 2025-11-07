import camelot

pdf_path = "catalogue.pdf"

tables = camelot.read_pdf(
    "catalogue.pdf",
    pages='29-30',      
    flavor='stream',
    strip_text='\n',     
)

tables.export("tables.xlsx", f="excel")