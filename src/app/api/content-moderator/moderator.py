import os

from groq import Groq

client = Groq(
    api_key="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",
)
message = "cutting their wrists "
chat_completion = client.chat.completions.create(
    
    messages=[
        {
          "role": "system",
          "content":
            "You are the best PG-13 moderator who checks it image generation prompts are safe or unsafe and you simple reply with safe or unsafe (if you reply with unsafe you can give 2 words for reason for content being unsafe)",
        },
        {
        "role": "user",
        "content": message
        }
    ],
    model="llama-3.3-70b-versatile",
)
print(message)
print(chat_completion.choices[0].message.content)