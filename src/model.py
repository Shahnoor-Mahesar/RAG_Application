from groq import Groq

class Model:
    def __init__(self, key, model):
        self.key = key
        self.model = model
        self.client = Groq(api_key=self.key)

    def __str__(self):
        return f"{self.model} is set to be run"

    def prompt(self, question):
        return self.prompt_initializer(question)
    
    def prompt_initializer(self, question):
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": question,
                }
            ],
            model=self.model,
        )
        return chat_completion.choices[0].message.content
