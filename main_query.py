from rag.rag_retriever import RAGRetriever
from rag.rag_llm import RAGLLM
from rag.query_enhancer import QueryEnhancer

def main():
    retriever = RAGRetriever("data/metadata.jsonl")
    llm = RAGLLM()
    enhancer = QueryEnhancer()

    while True:
        query = input("Query: ")
        query = enhancer.enhance(query)

        context = retriever.search(query)
        answer = llm.generate(query, context)

        print("\nAnswer:", answer, "\n")

if __name__ == "__main__":
    main()