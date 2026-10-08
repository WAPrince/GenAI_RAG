import boto3

from pycharmai.config.settings import settings

class BedrockLLM:

    def __init__(self) -> None:

        self.client = boto3.client(
            "bedrock-runtime",
            region_name=settings.aws_region,
        )


    def ask(self, prompt: str) -> str:

        response = self.client.converse(
            modelId=settings.bedrock_model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "text": prompt
                        }
                    ],
                }
            ],
        )
        return response["output"]["message"]["content"][0]["text"]