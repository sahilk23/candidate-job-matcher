#  Candidate job matcher

This project is a simple yet effective **Candidate Matcher System** that matches a given job description with the most relevant candidate profiles. It uses **TF-IDF vectorization** and **cosine similarity** to find candidates whose skills, projects, and profile comments closely match the job requirements.

##  Objective

To assist recruiters in filtering and shortlisting candidates by matching job descriptions against a pool of candidate data using natural language processing techniques.


##  Features

- Match job descriptions with candidate profiles using **TF-IDF**.
- Ranks candidates based on **cosine similarity** score.
- Reads candidate data from a CSV file.
- Prints the top 10 most relevant candidates.

##  Tech Stack

- **Language:** Python
- **Libraries:** 
  - `pandas` for data manipulation
  - `scikit-learn` for vectorization and similarity
- **Algorithm:** 
  - `TfidfVectorizer` for feature extraction
  - `cosine_similarity` for measuring relevance

---

##  How It Works

1. **Input:** A job description (entered via command line).
2. **Data Source:** A CSV file (`candidates.csv`) containing:
   - `Job Skills`
   - `Projects`
   - `Comments`
3. **Processing:**
   - Combines the above fields into a single string per candidate.
   - Vectorizes the candidate data and job description using `TF-IDF`.
   - Calculates cosine similarity between the job description and each candidate.
4. **Output:** Displays the top 10 matching candidate profiles.




##  Dataset 

The dataset `candidates.csv` contains structured information about multiple job candidates. It serves as the core input for matching with job descriptions.


| Column Name     | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| **Name**        | Full name of the candidate.                                                 |
| **Job Skills**  | List of technical or professional skills the candidate possesses.          |
| **Projects**    | Projects the candidate has worked on, typically aligned with their skills. |
| **Comments**    | Additional remarks, strengths, or summary notes about the candidate.       |

This dataset is used to build a textual representation of each candidate’s profile, which is then compared to a given job description using **TF-IDF** and **cosine similarity** for ranking relevance.
