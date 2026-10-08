import boto3

class BedrockClient:
    def __init__(self, region: str):
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region,
        )

    def converse(self, model_id: str, prompt: str):

        response = self.client.converse(
            modelId=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "text": "what date is today?"  # prompt
                        }
                    ],
                }
            ],
        )
        return response