from openai import OpenAI
import os

client = OpenAI()
path = "../../datasets/style_theme_anglais"
name_exercices = "A1_Queneau_anglais"
name_feneon = "B1_Feneon_anglais"
exercices = os.listdir(f"{path}/{name_exercices}")
nouvelles = os.listdir(f"{path}/{name_feneon}")
# nouvelles = os.listdir("./a_transformer/Feneon")

for i in range(len(exercices)):
    print(exercices[i])
    exercice = open(f"{path}/{name_exercices}/{exercices[i]}", "r")
    nouvelle = open(f"{path}/{name_feneon}/{nouvelles[i]}", "r")

    prompt = "Re write this text :\n" + nouvelle.read() + "\n Copying the style of this second text :" + exercice.read()
    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a creative AI assistant for content creation and writing."},
        {"role": "user", "content": prompt}
    ]
    )
    exercice.close()
    nouvelle.close()

    with open(f"{path}/generated/{exercices[i]}", "x") as f:
        f.write(completion.choices[0].message.content)