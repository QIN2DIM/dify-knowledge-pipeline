import json
import os
from abc import ABC, abstractmethod
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import tiktoken
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter, Language
from loguru import logger
from tqdm import tqdm

from dify_knowledge_pipeline.fire_drop import DifyFireDrop

SEPARATOR = "\n\n------------\n\n"

MAX_TOKENS = 4096


def normalize_path(path: str | os.PathLike | Path) -> Path:
    if isinstance(path, Path):
        return path
    if isinstance(path, os.PathLike):
        return Path(path.__fspath__())
    if isinstance(path, str):
        return Path(path)
    raise TypeError(f"Unsupported path type: {type(path)}")


def clean_mdx_schema_info(text) -> dict | None:
    with suppress(Exception):
        m = {}
        for c in text.split("---", 2)[1].strip().split("\n"):
            k, v = c.strip().split(":")
            v = v.replace('"', " ").strip()
            m[k.strip()] = v.strip()
        return m


def fork_tech_docs_markdown_to_chunks(
    fdr_docs: Path | str | os.PathLike,
    fdr_out: Path | str | os.PathLike,
    *,
    encoding_name: str = "gpt2",
    chunk_size: int = 4096,
    chunk_overlap_ratio: float = 0.15,
    **kwargs,
):
    """
    技术文档风格的语料嵌入

    - max_tokens <= 1000

    Args:
        fdr_out:
        fdr_docs:
        encoding_name: ['gpt2', 'r50k_base', 'p50k_base', 'p50k_edit', 'cl100k_base', 'o200k_base']
        chunk_size:
        chunk_overlap_ratio:

    Returns:

    """
    fdr_docs = normalize_path(fdr_docs)
    fdr_out = normalize_path(fdr_out)

    encoding = tiktoken.get_encoding(encoding_name)
    focus_ext = kwargs.get("ext", "*.md")

    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=True)

    # 文档文件作为一个独立的 embed 对象
    for fp in tqdm(fdr_docs.rglob(focus_ext), desc="splitting", postfix="embedding"):
        header_1_title = ""
        segments = []

        # ｛｛# 数据分片规则 #｝｝
        if not (text := fp.read_text(encoding="utf8").strip()):
            continue

        text = text.replace("\n\n", "\n")

        # 1. IF 源文档总长度 max_tokens < MAX_TOKENS，无需分块直接嵌入
        # 去掉过短的片段，切分过长的片段
        num_tokens = len(encoding.encode(text))
        if num_tokens < 50:
            continue
        if num_tokens < chunk_size:
            segments.append(text)

        mdx_schema_info = clean_mdx_schema_info(text) if focus_ext == "*.mdx" else {}

        # 2. 自定义的分块规则
        md_header_splits = markdown_splitter.split_text(text)
        for i, doc in enumerate(md_header_splits):
            metadata = doc.metadata
            content = doc.page_content.strip()

            if not header_1_title:
                # 将 FIRST 标题设为文件名
                for h in [1, 2, 3, 4]:
                    if header := metadata.get(f"Header {h}"):
                        header_1_title = header
                        break

            if metadata:
                # 格式化 Q&A
                metadata_str = " / ".join(list(metadata.values()))
                mdx_schema_info.update({"section": metadata_str, "content": content})
                segment = json.dumps(mdx_schema_info, ensure_ascii=False)
            else:
                # 无法自动解析 Question，则仅存储文本块
                metadata_str = ""
                segment = content

            if (num_tokens := len(encoding.encode(segment))) < MAX_TOKENS:
                if num_tokens < 50 and ("toc: menu" in segment or "toc: content" in segment):
                    continue
                if num_tokens < 50 and not metadata_str:
                    continue
                # 如果 Q&A 问答对符合 max_tokens 长度规范，无需进一步预处理
                segments.append(segment)
                continue

            # 拟合块状态，动态调整参数
            if metadata_str:
                _tmp = mdx_schema_info.copy()
                _tmp["content"] = ""
                _segment_tmp = json.dumps(_tmp, ensure_ascii=False)
                schema_num_tokens = len(encoding.encode(_segment_tmp))
                fixed_chunk_size = int((chunk_size - schema_num_tokens) * 0.98)
            else:
                fixed_chunk_size = MAX_TOKENS
            chunk_overlap = int(fixed_chunk_size * chunk_overlap_ratio)
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=encoding_name, chunk_size=fixed_chunk_size, chunk_overlap=chunk_overlap
            )

            # 切分过长的块，保持结构化切片
            chunks = text_splitter.split_text(content)
            for sid, chunk in enumerate(chunks):
                chunk = chunk.strip()
                if metadata_str:
                    mdx_schema_info.update({"section": metadata_str, "content": chunk})
                    chunk = json.dumps(mdx_schema_info, ensure_ascii=False)
                segments.append(chunk)
                _validate_max_tokens(encoding, chunk, fp.name, sid=i)

        # {{# 文件命名 #}}
        header_1_title = f"{fp.name}_{header_1_title}" if header_1_title else fp.name
        for ext_ in [".md", ".mdx"]:
            if header_1_title.endswith(ext_):
                header_1_title = header_1_title.replace(ext_, ".txt")

        yield _offload(header_1_title, segments, fp, fdr_out, prefix_name=kwargs.get("prefix_name"))


ts_block = """
// path = {path}

```ts
{code}
```
"""


def fork_source_code_ts_to_chunks(
    fdr_docs: Path | str | os.PathLike,
    fdr_out: Path | str | os.PathLike,
    *,
    encoding_name: str = "gpt2",
    chunk_size: int = 1500,
    chunk_overlap_ratio: float = 0.15,
    **kwargs,
):
    fdr_docs = normalize_path(fdr_docs)
    fdr_out = normalize_path(fdr_out)

    chunk_overlap = int(chunk_size * chunk_overlap_ratio)
    encoding = tiktoken.get_encoding(encoding_name)

    text_splitter = RecursiveCharacterTextSplitter.from_language(
        chunk_size=chunk_size, language=Language.TS, chunk_overlap=chunk_overlap
    )

    for fp in tqdm(fdr_docs.rglob("*.ts"), desc="splitting", postfix="embedding"):
        text = fp.read_text(encoding="utf-8")
        segments = []

        code_path = f"{fp.parent}\\{fp.name}"
        segment = ts_block.format(path=code_path, code=text)

        num_tokens = len(encoding.encode(segment))
        if num_tokens < chunk_size:
            segments.append(segment.strip())
            continue

        # ｛｛# 数据分片规则 #｝｝
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            segment = ts_block.format(path=code_path, code=chunk).strip()
            _validate_max_tokens(encoding, segment, fp.name)
            segments.append(segment)

        # {{# 文件命名 #}}
        header_1_title = fp.name
        for _ext in [".ts", ".tsx"]:
            if header_1_title.endswith(_ext):
                header_1_title = header_1_title.replace(_ext, ".txt")

        yield _offload(header_1_title, segments, fp, fdr_out, prefix_name=kwargs.get("prefix_name"))


def _offload(header_1_title: str, segments: List[str], fp: Path, fdr_out: Path, prefix_name=None):
    # 替换掉非法文件命名字符
    if not header_1_title.endswith(".txt"):
        header_1_title = f"{header_1_title}.txt"
    inv = {"\\", "/", ":", "*", "?", "<", ">", "|", "\n"}
    for i in inv:
        header_1_title = header_1_title.replace(i, "")

    # 将分片压缩到一个卡片中，存储至单个文件
    knowledge_card = SEPARATOR.join(segments)

    fp_name = f"{str(list(fp.parents)[0])}_{header_1_title}".replace("\\", "_")
    if prefix_name:
        fp_name = f"{prefix_name}_{fp_name}"

    # ｛｛# 数据存储 #｝｝
    fdr_out.mkdir(exist_ok=True, parents=True)
    fp_out = fdr_out / fp_name
    fp_out.write_text(knowledge_card, encoding="utf8")

    if knowledge_card:
        table_name = fp_name.replace(".txt", "")
        return table_name, knowledge_card


def _validate_max_tokens(encoding, segment: str, fp_name: str, *, max_tokens=MAX_TOKENS, sid=0):
    num_tokens_after_splitting = len(encoding.encode(segment))
    if num_tokens_after_splitting >= max_tokens:
        logger.warning(
            f"[{sid}] 块异常，max_tokens>={max_tokens} {len(segment)=} {num_tokens_after_splitting=} {fp_name=}"
        )
        return False
    return True


@dataclass
class KnowledgePipline(ABC):
    db_name: str
    sync_to_dify: bool = False
    force_override: bool = False
    separator = "\n\n------------\n\n"

    @abstractmethod
    def _invoke(self, **kwargs):
        raise NotImplementedError

    def invoke(self, sync_to_dify: bool = None):
        if sync_to_dify is not None:
            self.sync_to_dify = sync_to_dify

        self._invoke()
        return self

    def delete_all(self):
        dify_datasets = DifyFireDrop(separator=self.separator)
        dify_datasets.delete_all_document(db_name=self.db_name)

    def _sync_to_dify(self, table_to_knowledge: Dict[str, str]):
        if self.sync_to_dify and table_to_knowledge:
            dify_datasets = DifyFireDrop(separator=self.separator)
            dify_datasets.embed_knowledge(table_to_knowledge, db_name=self.db_name, force_override=self.force_override)
        return self
