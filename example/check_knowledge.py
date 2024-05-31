import dotenv

from dify_knowledge_pipeline import KnowledgeDatasetsClient

dotenv.load_dotenv()


class DatasetId:
    sample = "dbd22124-36cd-4849-bbc4-82d83093062e"


def main():
    """
    # DIFY_BASE_URL=http://localhost/v1
    # DIFY_KNOWLEDGE_API_KEY=dataset-xxx
    Returns:

    """
    kdc = KnowledgeDatasetsClient.from_env(dataset_id=DatasetId.sample)
    response = kdc.list_datasets()
    print(response.json())


if __name__ == "__main__":
    main()
