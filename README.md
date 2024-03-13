# Sentiment Analysis on Twitter Data

This project tackles the challenge of classifying Twitter data into positive or negative sentiments using both machine learning and deep learning techniques. The work is divided across four Jupyter notebooks, each dedicated to a different phase of the sentiment analysis process, ranging from data preprocessing to employing advanced modeling techniques.

## Project Structure

- `01_data_preprocessing.ipynb`: Initiates the project by loading, cleaning, and preprocessing the tweet data to prepare it for analysis.
- `02_EDA.ipynb`: Delves into exploratory data analysis (EDA) to unveil the dataset's underlying patterns, focusing on aspects like tweet length and the prevalence of certain words across sentiments.
- `03_modelling_and_evaluation.ipynb`: Implements and assesses the performance of traditional machine learning models such as Logistic Regression, LinearSVC, and BernoulliNB in classifying tweet sentiments.
- `04_advanced_models.ipynb`: Advances into exploring neural network architectures like Simple RNN, LSTM, and GRU, analyzing their capacity to enhance sentiment classification accuracy.

## Key Findings

- Advanced modeling techniques, particularly the GRU model, outperformed traditional machine learning approaches in accuracy, showcasing the potential of deep learning in capturing complex patterns in text data.
- The analysis provided insights into the importance of choosing the right model based on the dataset's characteristics and the specific task at hand.
- Traditional models remain competitive, with Logistic Regression leading in performance among them, proving the ongoing relevance of these approaches in sentiment analysis.

## Getting Started

Interested in running these analyses? Make sure Python 3.x is installed along with Jupyter Notebook or JupyterLab. Install all necessary libraries with:

```bash
pip install -r requirements.txt
