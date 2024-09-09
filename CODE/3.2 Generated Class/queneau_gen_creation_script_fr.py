from openai import OpenAI
import os

client = OpenAI()
path = "../../datasets/"
base = "A1_Queneau"
exercices = os.listdir(f"{path}/{base}")
# nouvelles = os.listdir("./a_transformer/Feneon")

for i in range(len(exercices)):
    print(exercices[i])
    exercice = open(f"{path}/{base}/{exercices[i]}", "r")
    # nouvelle = open(f"./a_transformer/Feneon/{nouvelles[i]}", "r")

    prompt = "Ré écris ce texte en copiant le style de Fénéon dans les 'nouvelles en trois lignes' :\n" + exercice.read() 
    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a creative AI assistant for content creation and writing in french."},
        {"role": "user", "content": prompt}
    ]
    )
    exercice.close()
    # nouvelle.close()

    with open(f"{path}/generated/{exercices[i]}", "x") as f:
        f.write(completion.choices[0].message.content)
