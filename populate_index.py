from symai.shellsv import retrieval_augmented_indexing
from symai.functional import EngineRepository
from symai.backend.engines.index.engine_pinecone import IndexEngine


def run():
    # Register embeddings engine globally for all Symbols from plugin
    EngineRepository.register_from_plugin('embedding', plugin='ExtensityAI/embeddings', kwargs={'model': 'all-mpnet-base-v2'}, allow_engine_override=True)
    EngineRepository.register('index', IndexEngine(index_name='dataindex',
                                                   index_dims=768,
                                                   index_top_k=5))
    # insert into the index
    retrieval_augmented_indexing('!src/evals/snippets', index_name='dataindex')


if __name__ == '__main__':
    run()