import streamlit as st
from keybert import KeyBERT

# Create the KeyBERT model
#model = KeyBERT("distilbert-base-nli-mean-tokens")
model = KeyBERT()

# Define a function to extract keywords
def extract_keywords(doc):
    # Check if the input document is null or empty
    if doc is None or doc.strip() == "":
        return []

    # Extract the keywords
    keywords = model.extract_keywords(doc)

    # Return the keywords
    return keywords

# Create the Streamlit app
def main():
    # Set the page title
    st.set_page_config(page_title="Keyword Extractor")

    # Add a title and a subtitle
    st.title("Keyword Extractor")
    st.markdown("Extract keywords from a document using a machine learning model.")

    # Add a file uploader
    file = st.file_uploader("Upload a document", type=["txt"])

    # If a file is uploaded
    if file is not None:
        # Read the file contents
        doc = file.read().decode("utf-8")

        # Extract the keywords
        keywords = model.extract_keywords(doc, keyphrase_ngram_range=(1, 2), 
                                         stop_words='english', use_mmr=True, diversity=0.6, top_n=15)
  

        # Display the keywords in a table
        if len(keywords) > 0:
            st.write(keywords)
            st.write('hello')
        else:
            st.write("No keywords found.")

# Run the app
if __name__ == "__main__":
    main()
