from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Dict, Any

import httpx
from loguru import logger

from dify_knowledge_pipline.models import Segment


@dataclass
class KnowledgeDatasetsClient:
    """
    http://192.168.1.180/datasets?category=api
    """

    api_key: str
    base_url: str = "http://192.168.1.180/v1"
    dataset_id: str = ""
    document_id: str = ""

    storage_dir = Path(".cache/knowledge/")

    def __post_init__(self):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        self.client = httpx.Client(base_url=self.base_url, headers=headers)

        self.storage_dir.mkdir(exist_ok=True, parents=True)

    @classmethod
    def from_env(
        cls,
        api_key: str | None = None,
        base_url: str | None = None,
        dataset_id: str = "",
        document_id: str = "",
    ):
        base_url = base_url or os.environ["DIFY_BASE_URL"]
        api_key = api_key or os.environ["DIFY_KNOWLEDGE_API_KEY"]
        return cls(
            api_key=api_key, base_url=base_url, dataset_id=dataset_id, document_id=document_id
        )

    def _cache_interface_response(self, response: httpx.Response, filename: str | None = None):
        if not filename or not isinstance(filename, str):
            return

        try:
            fp = self.storage_dir / filename
            fp.write_text(
                json.dumps(response.json(), ensure_ascii=False, indent=2), encoding="utf8"
            )
        except Exception as err:
            logger.error(f"Failed to save knowledge dataset to disk: {err}")

    def _send_request(
        self, request_method: str, url: str, *, files=None, json=None, params=None, **kwargs
    ) -> httpx.Response:
        dataset_id = kwargs.get("dataset_id", self.dataset_id)
        if not dataset_id and ("/datasets" != url) and (request_method != "GET"):
            raise ValueError("dataset_id must be specified")

        payload = json or kwargs.get("payload")
        response = self.client.request(
            request_method, url, files=files, json=payload, params=params
        )
        self._cache_interface_response(response, kwargs.get("cache_log"))

        return response

    def create_document_by_text(
        self,
        name: str,
        text: str,
        process_rule: Dict[str, Any] | None = None,
        indexing_technique: Literal["high_quality", "economy"] = "high_quality",
        *,
        dataset_id: str | None = "",
    ):
        """
        通过文本创建文档

        此接口基于已存在知识库，在此知识库的基础上通过文本创建新的文档
        Args:
            name: 文档名称
            text: 文档内容
            process_rule: 处理规则
                - mode (string) 清洗、分段模式 ，automatic 自动 / custom 自定义
                - rules (object) 自定义规则（自动模式下，该字段为空）
                    - pre_processing_rules (array[object]) 预处理规则
                        - id (string) 预处理规则的唯一标识符
                            - 枚举：
                                - remove_extra_spaces 替换连续空格、换行符、制表符
                                - remove_urls_emails 删除 URL、电子邮件地址
                        - enabled (bool) 是否选中该规则，不传入文档 ID 时代表默认值
                    - segmentation (object) 分段规则
                        - separator 自定义分段标识符，目前仅允许设置一个分隔符。默认为 \n
                        - max_tokens 最大长度 (token) 默认为 1000
            indexing_technique: 索引方式
            dataset_id: 知识库 ID

        Returns:

        """
        dataset_id = dataset_id or self.dataset_id
        process_rule = process_rule or {"mode": "automatic"}
        payload = {
            "name": name,
            "text": text,
            "process_rule": process_rule,
            "indexing_technique": indexing_technique,
        }
        url = f"/datasets/{dataset_id}/document/create_by_text"
        return self._send_request(
            "POST", url, payload=payload, cache_log="create_document_by_text.json"
        )

    def create_document_by_file(
        self, data: dict, file: str | Path | os.PathLike, *, dataset_id: str | None = ""
    ):
        """
        通过文件创建文档

        此接口基于已存在知识库，在此知识库的基础上通过文件创建新的文档
        Args:
            data:
                **original_document_id 源文档 ID （选填）**
                    用于重新上传文档或修改文档清洗、分段配置，缺失的信息从源文档复制
                    源文档不可为归档的文档
                    当传入 original_document_id 时，代表文档进行更新操作，process_rule 为可填项目，不填默认使用源文档的分段方式
                    未传入 original_document_id 时，代表文档进行新增操作，process_rule 为必填
                **indexing_technique 索引方式**
                    high_quality 高质量：使用 embedding 模型进行嵌入，构建为向量数据库索引
                    economy 经济：使用 Keyword Table Index 的倒排索引进行构建
                **process_rule 处理规则**
                    mode (string) 清洗、分段模式 ，automatic 自动 / custom 自定义
                    rules (object) 自定义规则（自动模式下，该字段为空）
                        pre_processing_rules (array[object]) 预处理规则
                            id (string) 预处理规则的唯一标识符
                                枚举：
                                    remove_extra_spaces 替换连续空格、换行符、制表符
                                    remove_urls_emails 删除 URL、电子邮件地址
                            enabled (bool) 是否选中该规则，不传入文档 ID 时代表默认值
                        segmentation (object) 分段规则
                            separator 自定义分段标识符，目前仅允许设置一个分隔符。默认为 \n
                            max_tokens 最大长度 (token) 默认为 1000
            file: 需要上传的文件。
            dataset_id:

        Returns:

        """

    def create_datasets(self, name: str):
        """
        创建空知识库

        Args:
            name: 知识库名称

        Returns:

        """
        payload = {"name": name}
        return self._send_request(
            "POST", f"/datasets", json=payload, cache_log="create_datasets.json"
        )

    def create_segments(
        self, document_id: str, segments: List[Segment], *, dataset_id: str | None = ""
    ):
        """
        新增分段

        Args:
            document_id: 文档 ID
            segments: 分段信息
            dataset_id: 知识库 ID

        Returns:

        """

    def list_datasets(self, page: str = "1", limit: str = "20") -> httpx.Response:
        """
        知识库列表

        curl --location --request GET 'http://192.168.1.180/v1/datasets?page=1&limit=20' \
        --header 'Authorization: Bearer {api_key}'
        Args:
            page: 页码
            limit: 返回条数，默认 20，范围 1-100

        Returns:

        """
        payload = {"page": page, "limit": limit}
        return self._send_request(
            "GET", "/datasets", payload=payload, cache_log="list_datasets.json"
        )

    def list_documents(
        self,
        keyword: str | None = "",
        page: str | None = "",
        limit: str | None = "",
        *,
        dataset_id: str | None = "",
    ) -> httpx.Response:
        """
        知识库文档列表

        ```
        curl --location --request GET 'http://192.168.1.180/v1/datasets/{dataset_id}/documents' \
        --header 'Authorization: Bearer {api_key}'
        ```

        Args:
            dataset_id: 知识库 ID
            keyword: 搜索关键词，可选，目前仅搜索文档名称
            page: 页码，可选
            limit: 返回条数，可选，默认 20，范围 1-100

        Returns:

        """
        dataset_id = dataset_id or self.dataset_id
        params = {"keyword": keyword, "page": page, "limit": limit}
        urlpath = f"/datasets/{dataset_id}/documents"
        return self._send_request(
            "GET", urlpath, params=params, dataset_id=dataset_id, cache_log="list_documents.json"
        )

    def list_segments(
        self,
        document_id: str,
        keyword: str | None = None,
        status: str | None = "completed",
        *,
        dataset_id: str | None = "",
    ) -> httpx.Response:
        """
        查询文档分段

        ```
        curl --location --request GET 'http://192.168.1.180/v1/datasets/{dataset_id}/documents/{document_id}/segments' \
        --header 'Authorization: Bearer {api_key}' \
        --header 'Content-Type: application/json'
        ```

        Args:
            dataset_id: 知识库 ID
            document_id: 文档 ID
            keyword: 搜索关键词，可选
            status: 搜索状态，completed

        Returns:

        """
        dataset_id = dataset_id or self.dataset_id
        params = {"keyword": keyword, "status": status}
        urlpath = f"/datasets/{dataset_id}/documents/{document_id}/segments"
        return self._send_request(
            "GET", urlpath, params=params, dataset_id=dataset_id, cache_log="list_segments.json"
        )

    def get_documents_indexing_status(self, dataset_id: str, batch: str):
        """
        获取文档嵌入状态（进度）

        ```
        curl --location --request GET 'http://192.168.1.180/v1/datasets/{dataset_id}/documents/{batch}/indexing-status' \
        --header 'Authorization: Bearer {api_key}'
        ```

        Args:
            dataset_id: 知识库 ID
            batch: 上传文档的批次号

        Returns:

        """
        dataset_id = dataset_id or self.dataset_id
        urlpath = f"/datasets/{dataset_id}/documents/{batch}/indexing-status"
        return self._send_request("GET", urlpath)

    def update_documents_by_text(
        self,
        document_id: str,
        name: str | None = "",
        text: str | None = "",
        process_rule: dict | None = None,
        *,
        dataset_id: str | None = "",
    ):
        """
        通过文本更新文档

        此接口基于已存在知识库，在此知识库的基础上通过文本更新文档
        Args:
            process_rule:
            text:
            name:
            dataset_id: 知识库 ID
            document_id: 文档 ID

        Returns:

        """

    def update_documents_by_file(
        self,
        document_id: str,
        file: Path | str | os.PathLike,
        name: str | None = "",
        process_rule: dict | None = None,
        *,
        dataset_id: str | None = "",
    ):
        """
        通过文件更新文档

        此接口基于已存在知识库，在此知识库的基础上通过文件更新文档的操作。

        ```
        curl --location --request POST 'http://192.168.1.180/v1/datasets/{dataset_id}/document/{document_id}/create_by_file' \
        --header 'Authorization: Bearer {api_key}' \
        --form 'data="{"name":"Dify","indexing_technique":"high_quality","process_rule":{"rules":{"pre_processing_rules":[{"id":"remove_extra_spaces","enabled":true},{"id":"remove_urls_emails","enabled":true}],"segmentation":{"separator":"###","max_tokens":500}},"mode":"custom"}}";type=text/plain' \
        --form 'file=@"/path/to/file"'
        ```

        Args:
            dataset_id: 知识库 ID
            document_id: 文档 ID
            file: 需要上传的文件
            name: 文档名称 （选填）
            process_rule: 处理规则（选填）

        Returns:

        """

    def update_segments(
        self, segment_id: str, segments: List[Segment], *, dataset_id: str | None = ""
    ):
        """
        更新文档分段

        ```
        curl --location --request POST 'http://192.168.1.180/v1/datasets/{dataset_id}/documents/{document_id}/segments/{segment_id}' \
        --header 'Authorization: Bearer {api_key}' \
        --header 'Content-Type: application/json'\
        --data-raw '{"segment": {"content": "1","answer": "1", "keywords": ["a"], "enabled": false}}'
        ```

        Args:
            dataset_id: 知识库 ID
            segment_id: 文档分段ID
            segments:

        Returns:

        """

    def delete_documents(self, document_id: str, *, dataset_id: str | None = ""):
        """
        删除文档
        Args:
            dataset_id: 知识库 ID
            document_id: 文档 ID

        Returns:

        """

    def delete_segments(self, segment_id: str, *, dataset_id: str | None = ""):
        """
        删除文档分段

        ```
        curl --location --request DELETE 'http://192.168.1.180/v1/datasets/{dataset_id}/documents/{document_id}/segments/{segment_id}' \
        --header 'Authorization: Bearer {api_key}' \
        --header 'Content-Type: application/json'
        ```

        Args:
            dataset_id: 知识库 ID
            segment_id: 文档分段ID

        Returns:

        """
