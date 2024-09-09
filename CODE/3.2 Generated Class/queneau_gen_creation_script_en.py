from openai import OpenAI
import os

client = OpenAI()
path = "../../datasets/style_theme_anglais"
base = "A1_Queneau_anglais"
exercices = os.listdir(f"{path}/{base}")
# nouvelles = os.listdir("./a_transformer/Feneon")

for i in range(len(exercices)):
    print(exercices[i])
    exercice = open(f"{path}/{base}/{exercices[i]}", "r")
    # nouvelle = open(f"./a_transformer/Feneon/{nouvelles[i]}", "r")

    prompt = "Re write this text in strictly less than 30 words and using only 1 to 3 sentences :\n" + exercice.read() + "\n Copying Feneon's style in 'novels in three lines'"
    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a creative AI assistant for content creation and writing."},
        {"role": "user", "content": prompt}
    ]
    )
    exercice.close()
    # nouvelle.close()

    with open(f"{path}/generated/{exercices[i]}", "x") as f:
        f.write(completion.choices[0].message.content)