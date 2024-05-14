from enum import Enum


class DifyClientError(Enum):
    KNOWLEDGE_400_NO_FILE_UPLOADED = ("no_file_uploaded", 400, "Please upload your file.")
    KNOWLEDGE_400_TOO_MANY_FILES = ("too_many_files", 400, "Only one file is allowed.")
    KNOWLEDGE_413_FILE_TOO_LARGE = ("file_too_large", 413, "File size exceeded.")
    KNOWLEDGE_415_UNSUPPORTED_FILE_TYPE = ("unsupported_file_type", 415, "File type not allowed.")
    KNOWLEDGE_400_HIGH_QUALITY_DATASET_ONLY = (
        "high_quality_dataset_only",
        400,
        "Current operation only supports 'high-quality' datasets.",
    )
    KNOWLEDGE_400_DATASET_NOT_INITIALIZED = (
        "dataset_not_initialized",
        400,
        "The dataset is still being initialized or indexing. Please wait a moment.",
    )
    KNOWLEDGE_403_ARCHIVED_DOCUMENT_IMMUTABLE = (
        "archived_document_immutable",
        403,
        "The archived document is not editable.",
    )
    KNOWLEDGE_409_DATASET_NAME_DUPLICATE = (
        "dataset_name_duplicate",
        409,
        "The dataset name already exists. Please modify your dataset name.",
    )
    KNOWLEDGE_400_INVALID_ACTION = ("invalid_action", 400, "Invalid action.")
    KNOWLEDGE_400_DOCUMENT_ALREADY_FINISHED = (
        "document_already_finished",
        400,
        "The document has been processed. Please refresh the page or go to the document details.",
    )
    KNOWLEDGE_400_DOCUMENT_INDEXING = (
        "document_indexing",
        400,
        "The document is being processed and cannot be edited.",
    )
    KNOWLEDGE_400_INVALID_METADATA = (
        "invalid_metadata",
        400,
        "The metadata content is incorrect. Please check and verify.",
    )
