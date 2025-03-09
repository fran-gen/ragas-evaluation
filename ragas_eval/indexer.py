from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PDFIndexer:
    def __init__(self, pdf_path: str, chunk_size: int = 2048, chunk_overlap: int = 256):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def create_chunks(self):
        """Load the PDF and create chunks of text."""
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        for document in chunks:
            document.metadata['file_name'] = document.metadata['source']
        return chunks
        