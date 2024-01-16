from symai.shellsv import retrieval_augmented_indexing
from symai.functional import EngineRepository
from symai.backend.engines.index.engine_pinecone import PineconeIndexEngine
from symai.backend.engines.index.engine_vectordb import VectorDBIndexEngine


def run():
    # Register embeddings engine globally for all Symbols from plugin
    EngineRepository.register_from_plugin('embedding', plugin='ExtensityAI/embeddings', kwargs={'model': 'all-mpnet-base-v2'}, allow_engine_override=True)
    # EngineRepository.register('index', PineconeIndexEngine(index_name='dataindex',
    #                                                        index_dims=768,
    #                                                        index_top_k=5))
    vectorDB = VectorDBIndexEngine(index_name='dataindex',
                                   index_dims=768,
                                   index_top_k=5)
    EngineRepository.register('index', vectorDB)
    # insert into the index
    retrieval_augmented_indexing('!src/evals/snippets', index_name='dataindex')
    # # need to persist in-memory to disk
    vectorDB.save()


if __name__ == '__main__':
    run()