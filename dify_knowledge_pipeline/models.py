from __future__ import annotations

from enum import Enum
from typing import List, Dict, Any

from pydantic import BaseModel, Field


class Segment(BaseModel):
    content: str = Field(..., description="(text) 文本内容/问题内容，必填")
    answer: str | None = Field("", description="(text) 答案内容，非必填，如果知识库的模式为qa模式则传值")
    keywords: List[str] = Field(default_factory=list, description="(list) 关键字，非必填")


class Segmentation(BaseModel):
    separator: str = Field("\n", description="自定义分段标识符，目前仅允许设置一个分隔符。")
    max_tokens: int = Field(1000, description="最大长度 (token) 默认为 1000")


class PreProcessingRule(str, Enum):
    REMOVE_EXTRA_SPACES = "remove_extra_spaces"
    REMOVE_URLS_EMAILS = "remove_urls_emails"


class CustomRules(BaseModel):
    pre_processing_rules: List[Dict[str, Any]]
    segmentation: Segmentation


class ProcessRule(BaseModel):
    mode: str = Field(
        "automatic",
        description=" (string) 清洗、分段模式 ，automatic 自动 / custom 自定义",
        examples=["automatic", "custom"],
    )
    rules: CustomRules | None = Field(None, description="(object) 自定义规则（自动模式下，该字段为空）")
