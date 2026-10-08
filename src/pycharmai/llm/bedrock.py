from pycharmai.config.settings import settings
from langchain_aws import ChatBedrockConverse


def create_llm() -> ChatBedrockConverse:

    return ChatBedrockConverse(
        model=settings.bedrock_model_id,
        region_name=settings.aws_region,
        temperature=0,
        max_tokens=1000,
    )