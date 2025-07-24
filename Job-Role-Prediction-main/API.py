from flask import Flask, jsonify, request
import re
import pickle

clf = pickle.load(open('clf.pkl', 'rb'))
tfidfd = pickle.load(open('tfidf.pkl', 'rb'))

def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape(
        """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text


def getCategories(resume_text):

    cleaned_resume = clean_resume(resume_text)
    input_features = tfidfd.transform([cleaned_resume])
    prediction_id = clf.predict(input_features)[0]

    # Map category ID to category name
    category_mapping = {
        15: "Java Developer",
        23: "Testing",
        8: "DevOps Engineer",
        20: "SDE",
        24: "Web Designing",
        12: "HR",
        13: "Hadoop",
        3: "Blockchain",
        10: "ETL Developer",
        18: "Operations Manager",
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
        21: "SAP Developer",
        5: "Civil Engineer",
        0: "Advocate",
    }

    category_name = category_mapping.get(prediction_id, "Unknown")



    return category_name

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if (request.method == 'GET'):
        # data = request.get_data()
        # print(data)
        data=request.form['data']
        print("data")
        category=getCategories(data)

        return jsonify({'data': category})


if __name__ == '__main__':
    app.run(debug=True)
