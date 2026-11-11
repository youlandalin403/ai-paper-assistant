from dotenv import load_dotenv
load_dotenv()

import os
from openai import OpenAI

api_key = os.environ.get("DEEPSEEK_API_KEY") 
if not api_key:
    raise ValueError("请先设置 DEEPSEEK_API_KEY 环境变量")
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

contents=input("You: ")
messages = [{"role": "user", "content": contents}]
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        messages[0]
    ],
    stream=False
)

print(messages)
print("Assistant: ",response.choices[0].message.content)
print(type(response))
print(response)
print(response.model)
print(response.choices)
print(response.choices[0])
print(response.choices[0].message)
print(response.choices[0].message.content)
print(response.usage)

print(response.model_dump())