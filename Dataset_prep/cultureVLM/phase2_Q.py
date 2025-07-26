from huggingface_hub import InferenceClient
import pandas as pd
import re
import time

data = pd.read_csv("Dataset_CultureVLM_style - Concepts Introductions.csv")
questions_phase2 = pd.DataFrame()
questions_phase2['concept'] = data['concept']
print(questions_phase2.head())
#print(data.head())


'''
Assume you are given a public image showing the "{concept}" of {country}. Generate a multiple-choice
question based on this image and the introduction of {concept}. Provide the correct
answer immediately following the question.
Ensure the question delves into deeper cultural knowledge but does not directly name
the {concept}. The options should be somewhat confusing to increase the difficulty,
but there must be only one correct answer. Users can only answer based on the image,
so don’t mention any "introduction" or "{concept}" in the question. Use the following
format for your generated question:
- Question: [Your Question] Options: (A) [Option 1] (B) [Option 2] (C) [Option 3] (D)
[Option 4]
- Answer: (X) [Option X]
Here are two examples:
Image: Peking Duck
Introduction of Peking Duck: Peking Duck is a famous Chinese dish that originated in
Beijing during the Imperial era. The dish dates back to the Yuan Dynasty (1271-1368)
and became a staple in the Ming Dynasty (1368-1644). Traditionally, Peking Duck is
known for its thin, crispy skin and is served with pancakes, hoisin sauce, and
scallions. The preparation involves inflating the duck to separate the skin from the
fat, marinating it, and roasting it in a closed or hung oven. It is considered a
national dish of China and a symbol of Chinese culinary art.
- Question: During which dynasty did the dish shown in the image become a staple in
the cuisine of its country? Options: (A) Tang Dynasty (B) Song Dynasty (C) Ming
Dynasty (D) Qing Dynasty
- Answer: (C) Ming Dynasty
Image: The White House
Introduction of The White House: The White House, located at 1600 Pennsylvania Avenue
NW in Washington, D.C., is the official residence and workplace of the President of
the United States. Construction began in 1792 and was completed in 1800. The building
was designed by Irish-born architect James Hoban in the neoclassical style. It has
been the residence of every U.S. president since John Adams. The White House has
undergone several renovations and expansions, including the addition of the West Wing
and the Oval Office. It is a symbol of the U.S. government and a site of significant
historical events.
- Question: Who was the architect responsible for designing the building shown in the
image? Options: (A) James Hoban (B) Benjamin Latrobe (C) Thomas Jefferson (D) Charles
Bulfinch
- Answer: (A) James Hoban
Now please generate the question for the Image: {concept} of {country}
{introduction}
'''

client = InferenceClient(
	provider="hf-inference",
	api_key=""
)

pattern = r"</think>([\s\S]*)"
# msg = '<think>\nOkay, so I need to generate a multiple-choice question based on the image of the Menkaure Pyramid of Egypt. The question should delve into deeper cultural knowledge but without directly mentioning the pyramid. The options should be somewhat confusing but only have one correct answer. Users can only answer based on the image, so no introduction can be included.\n\nFirst, I should analyze the information given about the Menkaure Pyramid. It\'s the smallest of the three main pyramids at Giza, built during the Fourth Dynasty of the Old Kingdom. It\'s made with granite in the lower sections, which is different from the usual limestone. It\'s the tomb of Pharaoh Menkaure, part of a complex including a mortuary temple, causeway, and subsidiary pyramids. It was left unfinished and completed by his successor.\n\nNow, to create the question, I need to think of something that requires knowledge beyond just recognizing the pyramid. Considering it\'s the smallest, maybe focusing on its unique features relative to other pyramids. Or perhaps its construction details.\n\nLooking at the example questions, they often ask about the dynasty, architect, or specific instrument used in construction. Since the Menkaure Pyramid is unique for using granite, maybe something about the material used in its lower sections.\n\nWait, the user example about the White House asked about the architect. So perhaps asking about an architectural feature or a fact about its construction. Alternatively, something about its size relative to others.\n\nBut the options need to be confusing. Maybe options about the number of subsidiary structures or the material used.\n\nWait, in the introduction, there\'s a mention that it\'s the smallest, and it used granite. So maybe the question is about its unique outer material, which is granite. Alternatively, its relation to Pharaoh Menkaure.\n\nBut the question should not name the pyramid, so perhaps the question is about the material of some structure in the complex.\n\nAlternatively, perhaps the question can be about when it was built, but that might be too direct. Since it\'s the fourth dynasty, maybe.\n\nWait, in the example, the question about the dish was about the dynasty, so similar approach could be used here.\n\nBut the question must focus on the pyramid image, so maybe not about the dynasty but about its construction.\n\nLooking again, the image shows the pyramid, so the question can be about what makes it unique in materials or structure.\n\nThe introduction mentions that the lower sections were clad in granite. So perhaps the question is about the material used in the lower sections.\n\nAlternatively, since it\'s the smallest, maybe a question about its size compared to others, but I think the user might want to focus on a more detailed feature.\n\nAlternatively, the question could be about something like the construction reason or the part that was left unfinished. But perhaps that\'s too vague.\n\nWait, the introduction also mentions that the pyramid complex includes a mortuary temple, a causeway, and three smaller subsidiaries. So maybe the question is about something associated with its structures.\n\nAlternatively, the time period or the successor who finished it.\n\nBut given the focus on the image, the best approach might be to focus on a unique feature visible in the image, such as the material.\n\nSo the question could be: "The structure shown in the image is notable for what unique feature of its construction?" Options would be about the material used, the number of queens\' pyramids, the dynasty it was built in, etc.\n\nAlternatively, the question could be about the material used in its lower sections, as that\'s unique.\n\nSo, to make it precise, perhaps the question is: "What material was primarily used for the lower sections of the structure shown in the image?"\n\nOptions: (A) Limestone (B) Sandstone (C) Granite (D) Marble.\n\nThe correct answer would be (C) Granite, since the introduction states that the lower sections were clad in granite.\n\nThis way, the question requires knowledge of the pyramid\'s construction materials without directly naming it. The options are somewhat confusing since three are types of stones, but only one is correct.\n\nAnother angle could be about the pyramid\'s size, asking which pyramid it is, but that\'s more direct and perhaps less deep.\n\nWait, but the user\'s example for the dish used dynasty, so maybe for the pyramid, it could use the material. Alternatively, the number of subsidiary pyramids.\n\nWait, the introduction says three smaller subsidiary pyramids, so the question could be: "How many subsidiary pyramids are part of the complex shown in the image?" Options (A) 2 (B) 3 (C) 4 (D) 5. Correct answer would be (B) 3.\n\nBut considering the image, which shows the main pyramid, would the subsidiary pyramids be visible enough? Maybe not, as the main image focuses on the pyramid itself. So the user might not be able to see the subsidiary structures, making that question harder to answer based solely on the image.\n\nTherefore, the best approach is to focus on a visible feature such as material. So, the question about the lower sections being made of granite seems appropriate.\n\nSo, finalizing the question as: "What material was primarily used for the lower sections of the structure shown in the image?" with options (A) Limestone (B) Sandstone (C) Granite (D) Marble. Answer is (C) Granite.\n\nAlternatively, considering the user\'s previous examples, sometimes the questions are about relevant historical or cultural facts tied to the image.\n\nAlternatively, perhaps about the pharaoh it\'s dedicated to, but that might be too direct, but since the introduction gives that info, but the question shouldn\'t mention Menkaure.\n\nWait, the example didn\'t mention the dish\'s name but referred to it as "the dish." Similarly, here, the question refers to "the structure shown in the image."\n\nAlternatively, to make it more challenging, the question could be about the pyramid\'s position relative to others or something like that.\n\nBut the key is to ensure that the answer is based on the image and the intro, but without giving away the specific pyramid. So focusing on the unique materials or other features.\n\nSo, I think the material-based question is solid because it\'s unique to Menkaure\'s pyramid.\n</think>\n\n**Question:**  \nWhat material was primarily used for the lower sections of the structure shown in the image?  \n**Options:** (A) Limestone (B) Sandstone (C) Granite (D) Marble  \n\n**Answer:**  \n(C) Granite'
# match = re.search(pattern,msg)
# print(match.group(1))
questions = []

for concept in data['concept']:

    introduction = data.loc[data['concept'] == concept, 'intro']
    introduction = introduction.to_list()[0]


    messages = [
        {
            "role": "user",
            "content": f"Assume you are given a public image showing the {concept} of Egypt. Generate a multiple-choice\
                        question based on this image and the introduction of {concept}. Provide the correct\
                        answer immediately following the question.\
                        Ensure the question delves into deeper cultural knowledge but does not directly name\
                        the {concept}. The options should be somewhat confusing to increase the difficulty,\
                        but there must be only one correct answer. Users can only answer based on the image,\
                        so don’t mention any introduction or {concept} in the question. Use the following\
                        format for your generated question:\
                        - Question: [Your Question] Options: (A) [Option 1] (B) [Option 2] (C) [Option 3] (D)\
                        [Option 4]\
                        - Answer: (X) [Option X]\
                        Here are two examples:\
                        Image: Peking Duck\
                        Introduction of Peking Duck: Peking Duck is a famous Chinese dish that originated in\
                        Beijing during the Imperial era. The dish dates back to the Yuan Dynasty (1271-1368)\
                        and became a staple in the Ming Dynasty (1368-1644). Traditionally, Peking Duck is\
                        known for its thin, crispy skin and is served with pancakes, hoisin sauce, and\
                        scallions. The preparation involves inflating the duck to separate the skin from the\
                        fat, marinating it, and roasting it in a closed or hung oven. It is considered a\
                        national dish of China and a symbol of Chinese culinary art.\
                        - Question: During which dynasty did the dish shown in the image become a staple in\
                        the cuisine of its country? Options: (A) Tang Dynasty (B) Song Dynasty (C) Ming\
                        Dynasty (D) Qing Dynasty\
                        - Answer: (C) Ming Dynasty\
                        Image: The White House\
                        Introduction of The White House: The White House, located at 1600 Pennsylvania Avenue\
                        NW in Washington, D.C., is the official residence and workplace of the President of\
                        the United States. Construction began in 1792 and was completed in 1800. The building\
                        was designed by Irish-born architect James Hoban in the neoclassical style. It has\
                        been the residence of every U.S. president since John Adams. The White House has\
                        undergone several renovations and expansions, including the addition of the West Wing\
                        and the Oval Office. It is a symbol of the U.S. government and a site of significant\
                        historical events.\
                        - Question: Who was the architect responsible for designing the building shown in the\
                        image? Options: (A) James Hoban (B) Benjamin Latrobe (C) Thomas Jefferson (D) Charles\
                        Bulfinch\
                        - Answer: (A) James Hoban\
                        Now please generate the question for the Image: {concept} of Egypt\
                        Introduction: {introduction}",
        }
    ]

    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", 
        messages=messages, 
        max_tokens=1500
    )

    question = completion.choices[0].message
    question = str(question)
    print(question)
    match = re.search(pattern, question)
    question = match.group(1)
    questions.append(question)
    time.sleep(60)

questions_phase2['questions'] = pd.Series(questions)
questions_phase2.to_csv("Questions_phase_2.csv")

#print("<think>\nAlright, so the user wants me to generate a multiple-choice question based on an image of the Sphinx of Egypt. They provided a detailed introduction, so I need to make sure I use that. The user emphasized that the question shouldn't directly mention the sphinx or the introduction but should delve into deeper cultural knowledge. The options should be somewhat confusing but with only one correct answer.\n\nFirst, I should think about key points from the introduction. The Sphinx is located near the Great Pyramids, it's a lion body with a human head, dates back to around 2500 BCE, and symbolizes strength and wisdom. It's a guardian of the pyramids and has historical and cultural significance.\n\nI need to create a question that doesn't name the Sphinx but uses these details. Maybe something about the purpose of the statue. The options should include other ancient structures, like the Parthenon, Colosseum, and others, to make it tricky.\n\nFor the correct answer, the introduction mentions it served as a guardian of the pyramids, so option C as a guardian of the pyramids would be correct.\n\nI should ensure the question is clear and only one option fits correctly. Let me structure it like the examples provided, making sure the options are plausible but only one is right.\n</think>\n\n- **Question:** What is the primary symbolism attributed to the ancient statue depicted in the image?  \nOptions:  \n(A) A representation of Greek democracy and its principles  \n(B) A symbol of Roman engineering and entertainment  \n(C) A guardian of the pyramids, embodying strength and wisdom  \n(D) A portrayal of the duality of human emotions  \n\n- **Answer:** (C) A guardian of the pyramids, embodying strength and wisdom")
'''
client = InferenceClient(
	provider="hf-inference",
	api_key="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"
)

messages = [
	{
		"role": "user",
		"content": "What is the capital of France?"
	}
]

completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", 
	messages=messages, 
	max_tokens=500
)

print(completion.choices[0].message)
'''