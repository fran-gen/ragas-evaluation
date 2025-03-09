from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import pandas as pd


class SyntheticDataGenerator:
    def __init__(self, generator_llm_model: str, critic_llm_model: str, chunks):
        self.generator_llm = ChatOpenAI(model=generator_llm_model)
        self.critic_llm = ChatOpenAI(model=critic_llm_model)
        self.embeddings = OpenAIEmbeddings()
        self.chunks = chunks
        self.generator = self.create_generator()

    def create_generator(self):
        """Creates a TestsetGenerator with the specified LLMs and embeddings."""
        return TestsetGenerator.from_langchain(
            self.generator_llm, self.critic_llm, self.embeddings
        )

    def generate_testset(
        self, test_size: int = 2, distributions: dict = None
    ) -> pd.DataFrame:
        """Generates a testset with the specified size and distributions."""
        if distributions is None:
            distributions = {simple: 1.0, reasoning: 0.0, multi_context: 0.0}

        testset = self.generator.generate_with_langchain_docs(
            self.chunks, test_size=test_size, distributions=distributions
        )
        return testset.to_pandas()
