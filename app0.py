import streamlit as st
import pandas as pd
import os
import subprocess

# Define the questions and options
questions = [
    {
        "question": "When you try to remember something, do you prefer to:",
        "options": {
            "Visualize it in your mind": "visual",
            "Talk through or listen to explanations": "auditory",
            "Write it down or use your hands": "kinesthetic"
        }
    },
    {
        "question": "What helps you learn best?",
        "options": {
            "Diagrams and charts": "visual",
            "Lectures and discussions": "auditory",
            "Hands-on activities and labs": "kinesthetic"
        }
    },
    {
        "question": "How do you prefer to spend your free time?",
        "options": {
            "Watching movies or reading": "visual",
            "Listening to music or podcasts": "auditory",
            "Engaging in sports or crafts": "kinesthetic"
        }
    }
]

# Function to calculate the learning style
def calculate_learning_style(responses):
    results = {"visual": 0, "auditory": 0, "kinesthetic": 0}
    for response in responses:
        results[response] += 1
    predominant_style = max(results, key=results.get)
    return predominant_style

# Function to create or load Excel file
def create_or_load_excel(filename):
    if os.path.exists(filename):
        return pd.read_excel(filename)
    else:
        return pd.DataFrame(columns=["Name", "School", "Grade", "Learning Style"])

# Function to save student details to Excel file
def save_to_excel(df, filename):
    df.to_excel(filename, index=False)

# Streamlit UI
def main():
    st.title("Learning Style Quiz")

    # Collect user information
    name = st.text_input("Enter your name")
    school = st.text_input("Enter your school name")
    grade = st.text_input("Enter your grade")

    # Store user responses
    responses = []
    # Display each question and options
    for question in questions:
        st.header(question["question"])
        for option, style in question["options"].items():
            if st.checkbox(option):
                responses.append(style)

    if not responses:
        st.warning("Please select at least one option for each question.")
    else:
        # Calculate learning style
        learning_style = calculate_learning_style(responses)

        # Save student details to DataFrame
        student_details = {"Name": [name], "School": [school], "Grade": [grade], "Learning Style": [learning_style]}

        # Save student details only after clicking submit button
        if st.button("Submit"):
            student_df = create_or_load_excel("student_details.xlsx")
            student_df = pd.concat([student_df, pd.DataFrame(student_details)], ignore_index=True)
            save_to_excel(student_df, "student_details.xlsx")
            st.success("Student details saved successfully!")

            subprocess.Popen(["streamlit", "run", "app6f.py"])
            st.stop()

if __name__ == "__main__":
    main()
