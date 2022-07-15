def clean_text(filename):
    import spacy
    import re

    nlp = spacy.load("en_core_web_sm")
    with open(filename, 'r') as f:
        text = f.read()

    doc = nlp(text)
    sentence_list = []
    for sent in doc.sents:
        if sent[0].is_title and sent[-1].is_punct:
            has_noun = 2
            has_verb = 1
            for token in sent:
                if token.pos_ in ["NOUN", "PROPN", "PRON"]:
                    has_noun -= 1
                elif token.pos_ == "VERB":
                    has_verb -= 1
            if has_noun < 1 and has_verb < 1:
                sentence_list.append(sent.text)

    for i in range(0, len(sentence_list)):
        sentence_list[i] = re.sub("\n", "", sentence_list[i])

    return sentence_list


print(clean_text("/home/marie/Downloads/sample_txt/Ahuja.txt"))
