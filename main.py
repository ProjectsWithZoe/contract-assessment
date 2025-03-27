import pdfplumber

#open pdf file that needs assessing
def openFile(pdf_file):
    text = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return '\n'.join(text)

pdf_text = openFile('CTAC.pdf')
print(pdf_text)