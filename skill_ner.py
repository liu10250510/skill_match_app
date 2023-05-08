import streamlit as st
import spacy
from spacy.matcher import PhraseMatcher
import pandas as pd

# load default skills data base
from skillNer.general_params import SKILL_DB
# import skill extractor
from skillNer.skill_extractor_class import SkillExtractor

# init params of skill extractor
nlp = spacy.load("en_core_web_lg")

def extract_skills(doc):

    # init skill extractor
    skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)

    annotations = skill_extractor.annotate(doc)

    # extract the matched skills to be a panda data 
    full_match_skills=[]
    full_match_scores=[]
    for skill in annotations['results']['full_matches']:
        full_match_skills.append(skill['doc_node_value'])
        full_match_scores.append(skill['score'])
    #print(skill['doc_node_value'], skill['score'])

    df_main = pd.DataFrame(list(zip(full_match_skills, full_match_scores)),
               columns =['Main_skill', 'Main_Skill_Score']).drop_duplicates()
    df_main.sort_values(by=['Main_Skill_Score'], ascending=False, inplace=True, ignore_index=True)

    ngram_scored_skills=[]
    ngram_scored_scores=[]
    for skill in annotations['results']['ngram_scored']:
        ngram_scored_skills.append(skill['doc_node_value'])
        ngram_scored_scores.append(skill['score'])
        #print(skill['doc_node_value'], skill['score'])
    df_ngram = pd.DataFrame(list(zip(ngram_scored_skills, ngram_scored_scores)),
               columns =['ngram_skill', 'ngram_Score']).drop_duplicates()
    df_ngram.sort_values(by=['ngram_Score'], ascending=False, inplace=True, ignore_index=True)
    return df_main, df_ngram

# Create the Streamlit app
def main():
    # Set the page title
    st.set_page_config(page_title="Skills Extractor")

    # Add a title and a subtitle
    st.title("Skills Extractor")
    st.markdown("Extract skills from a job description using a machine learning model.")

    # Add a file uploader
    file = st.file_uploader("Upload a document", type=["txt"])

    # If a file is uploaded
    if file is not None:
        # Read the file contents
        doc = file.read().decode("utf-8")

        # Extract the keywords
        main_skills, ngram_skills = extract_skills(doc)
  

        # Display the keywords in a table
        if main_skills.shape[0]>1 or ngram_skills.shape[0]>1:
            st.write(main_skills)
            st.write(ngram_skills)
        else:
            st.write("No Skills found.")

# Run the app
if __name__ == "__main__":
    main()
