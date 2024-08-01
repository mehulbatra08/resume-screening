import streamlit as st
import pickle
import re
import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data
# nltk.download('punkt')
nltk.download('stopwords')

# Loading Models for Resume Screening
try:
    clf = pickle.load(open('clf.pkl', 'rb'))
    tfidfd = pickle.load(open('tfidf.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
except Exception as e:
    st.error(f"An error occurred while loading models: {e}")

# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

# Load Our Dataset
@st.cache_data
def load_data(data):
    df = pd.read_csv(data)
    return df

# Function to vectorize text to cosine matrix
@st.cache_data
def vectorize_text_to_cosine_mat(data):
    count_vect = CountVectorizer()
    cv_mat = count_vect.fit_transform(data)
    cosine_sim_mat = cosine_similarity(cv_mat)
    return cosine_sim_mat

# Recommendation System
@st.cache_data
def get_recommendation(category, cosine_sim_mat, df, num_of_rec=10):
    # Filter courses based on the category
    category_courses = df[df['course_title'].str.contains(category, case=False)]
    
    if category_courses.empty:
        return pd.DataFrame()  # Return empty DataFrame if no courses found
    
    # Get the index of the first course in the category
    idx = category_courses.index[0]
    
    sim_scores = list(enumerate(cosine_sim_mat[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    selected_course_indices = [i[0] for i in sim_scores[1:num_of_rec+1]]
    selected_course_scores = [i[1] for i in sim_scores[1:num_of_rec+1]]
    
    result_df = df.iloc[selected_course_indices]
    result_df['similarity_score'] = selected_course_scores
    final_recommended_courses = result_df[['course_title', 'similarity_score', 'url', 'price', 'num_subscribers']]
    return final_recommended_courses

RESULT_TEMP = """
<div style="
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
">
    <h4 style="
        color: #0e1117;
        margin-bottom: 10px;
        font-size: 18px;
        font-weight: bold;
    ">{}</h4>
    <p style="margin-bottom: 5px;">
        <span style="color: #0e1117; font-weight: bold;">📈 Score:</span>
        <span style="color: #1f77b4;">{}</span>
    </p>
    <p style="margin-bottom: 5px;">
        <span style="color: #0e1117; font-weight: bold;">🔗 </span>
        <a href="{}" target="_blank" style="color: #1f77b4; text-decoration: none;">Course Link</a>
    </p>
    <p style="margin-bottom: 5px;">
        <span style="color: #0e1117; font-weight: bold;">💲 Price:</span>
        <span style="color: #1f77b4;">{}</span>
    </p>
    <p style="margin-bottom: 0;">
        <span style="color: #0e1117; font-weight: bold;">👨‍🎓 Students:</span>
        <span style="color: #1f77b4;">{}</span>
    </p>
</div>
"""

def main():
    st.title("Resume Screening and Course Recommendation App")

    uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = cleanResume(resume_text)
        input_features = tfidfd.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]
        
        category_mapping = {
            15: "HR",
            23: "Accounting Manager",
            8: "DevOps Engineer",
            20: "Operations Manager",
            24: "Web Designing",
            12: "Java Developer",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Python Developer",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SQL Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }
        category_name = category_mapping.get(prediction_id, "Unknown")
        st.write("Predicted Category: ", category_name)

        # Load the dataset
        df = load_data("udemy_course_data.csv")
        cosine_sim_mat = vectorize_text_to_cosine_mat(df['course_title'])

        # Get recommendations based on the predicted category
        y = category_name.split(' ')
        results = get_recommendation(y[0], cosine_sim_mat, df, 7)

        if results.empty:
            st.write(f"No courses found for the category: {category_name}")
        else:
            st.subheader("Recommended Courses:")
            for _, row in results.iterrows():
                rec_title = row['course_title']
                rec_score = row['similarity_score']
                rec_url = row['url']
                rec_price = row['price']
                rec_num_sub = row['num_subscribers']
                st.markdown(RESULT_TEMP.format(rec_title, rec_score, rec_url, rec_price, rec_num_sub), unsafe_allow_html=True)

if __name__ == '__main__':
    main()