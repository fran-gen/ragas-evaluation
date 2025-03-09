from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings


class BaseRAGPipeline:
    def __init__(self, llm, chunks):
        self.chunks = chunks
        self.embedding = OpenAIEmbeddings()
        self.model = ChatOpenAI(model_name=llm)
        self.vectorstore = self.create_vectorstore()
        self.retriever = self.create_retriever()
        self.prompt = self.create_prompt_template()
        self.rag_chain = self.create_rag_chain()

    def create_vectorstore(self):
        return Chroma.from_documents(self.chunks, self.embedding)

    def create_retriever(self):
        raise NotImplementedError("create_retriever method needs to be implemented by subclasses")

    def create_prompt_template(self):
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        return PromptTemplate(template=template, input_variables=["context", "question"])

    def create_rag_chain(self):
        return (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.model
            | StrOutputParser()
        )

    def run(self, question: str):
        answer = self.rag_chain.invoke(question)
        return answer

    def get_context(self, question: str):
        relevant_docs = self.retriever.get_relevant_documents(question)
        contexts = [doc.page_content for doc in relevant_docs]
        return contexts #"\n".join(contexts)

class SimpleRAGPipeline(BaseRAGPipeline):
    def create_retriever(self):
        return self.vectorstore.as_retriever()

class MultiQueryPipeline(BaseRAGPipeline):
    def create_retriever(self):
        return MultiQueryRetriever.from_llm(
            retriever=self.vectorstore.as_retriever(), llm=self.model
        )
