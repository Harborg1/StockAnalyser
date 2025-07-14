import os
from openai import OpenAI
from dotenv import load_dotenv

# Load your .env file
load_dotenv("passcodes.env")

# Create client using the API key from the environment variable
client = OpenAI(api_key=os.getenv("CHAT_GPT_KEY"))

# Initialize chat history
messages = [{"role": "system", "content": "You are a helpful assistant."}]

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    messages.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    reply = response.choices[0].message.content
    print("ChatGPT:", reply)

    messages.append({"role": "assistant", "content": reply}) 