import os
from dotenv import load_dotenv
import json
import asyncio
import httpx
from pathlib import Path

load_dotenv()

class CustomAPI:

    def __init__(self):
        self._base_url = f"http://{os.getenv('IP_ADDRESS')}"
    
    async def create_agent_run(
        self,
        message: str,
        agent: str
    ):
        url = f'{self._base_url}/agents/{agent}/runs/'
        headers = {
            'accept': 'application/json'
        }
        form_data = {
            'message': message,
            'stream': 'false',
            'background': 'false'
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, data=form_data, timeout=None)

        return response.json()



# Example usage:
if __name__ == '__main__':
    customai = CustomAPI()
    response = asyncio.run(customai.create_agent_run("how to patent a project?", os.getenv('AGENT')))

    test_dir = Path('observability/test')
    test_dir.mkdir(parents=True, exist_ok=True)
    with open(test_dir / 'response.json', 'w') as f:
        json.dump(response, f, indent=4)