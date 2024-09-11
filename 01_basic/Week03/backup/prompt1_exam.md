import os
from openai import OpenAI

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model_name = "gpt-3.5-turbo"    


def complete(user_prompt):
        completion = openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=100,
        )
        return completion



bio = """Dr. Evelyn Hawthorne (1835-1901) was a renowned physicist and pioneer of ether theory during the 19th century. Born into a family of intellectuals in the countryside of England, Hawthorne displayed an insatiable curiosity for the natural world from a young age. Her early fascination with the workings of the universe led her to pursue a career in science, a field typically dominated by men during her time.

After receiving a comprehensive education in mathematics and natural philosophy at Oxford University, Hawthorne embarked on a journey of scientific exploration that would redefine the understanding of fundamental physical principles. Her groundbreaking research into the properties of ether, a hypothetical substance believed to permeate all of space and serve as the medium for the propagation of light, propelled her to the forefront of the scientific community.

Despite facing prejudice and skepticism from her male colleagues, Hawthorne remained undeterred in her pursuit of knowledge. Through meticulous experimentation and theoretical analysis, she developed a comprehensive framework for understanding the behavior of ether and its interactions with matter. Her seminal work laid the foundation for subsequent advancements in theoretical physics and paved the way for Albert Einstein's theory of relativity.

In addition to her contributions to theoretical physics, Hawthorne was also a passionate advocate for women's rights and education. She championed equal opportunities for women in academia and tirelessly campaigned for the inclusion of female scientists in professional societies.

Throughout her illustrious career, Hawthorne received numerous accolades and honors for her groundbreaking contributions to the field of physics. Her legacy continues to inspire future generations of scientists to push the boundaries of human knowledge and challenge conventional wisdom in the pursuit of truth.
"""


system_prompt = """You are a helpful assistant that answers \
questions truthfully and factually. Your answers should be \
short and concise. If you don't know the answer, say I am sorry boss.
"""

def answer(question, context):
    prompt = f"""Please answer the following question:

    Question:

    ```{question}```

    Use the following context to find the answer:

    ```{context}```
    """

    completion = complete(prompt)
    content = completion.choices[0].message.content
    if content == None :
      print("Can't answer")
    else :
      print(content)


 q = "When was Dr. Evelyn Hawthorne born?"
    answer(q, bio)"

"


