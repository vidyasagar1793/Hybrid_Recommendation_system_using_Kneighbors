# Movie Recommendation System

Hybrid Movie recommendation system implemented in Python using the Streamlit library for the user interface and pandas, numpy, scikit-learn, and scipy for data manipulation and recommendation algorithms.

## Overview

The recommendation system is based on a hybrid approach combining collaborative filtering and content-based filtering techniques.

1. **Data Loading**: The system loads movie data from CSV files (`movie.csv` and `rating.csv`) containing information about movies and user ratings, respectively.

2. **Data Preprocessing**: The movie and rating data are preprocessed, including renaming columns and creating data structures for efficient processing.

3. **Recommendation Algorithm**:
   - The recommendation algorithm is implemented in the `HybridRecSys` class, which extends the `RecSys` class.
   - The `RecSys` class prepares the data and constructs a rating matrix for collaborative filtering.
   - The `HybridRecSys` class further enhances the recommendation system by incorporating movie features (genres) and using a nearest neighbors algorithm for content-based filtering.
   - Recommendations are generated based on similarity scores calculated between movies.

4. **User Interface**: The Streamlit library - simple web interface for users to interact with the recommendation system.
   - Users can input a movie title, and the system provides recommendations based on that input.
 Enter a movie title in the input field and click on the "Get Recommendations" button to see movie recommendations based on the entered title.

## File Structure

- `app.py`: Main script containing the Streamlit application and user interface.
- `movie.csv`: CSV file containing movie data.
- `rating.csv`: CSV file containing user ratings data.
- `README.md`: Documentation file providing an overview of the recommendation system.

## Dependencies

- pandas
- NumPy
- scikit-learn
- scipy
- streamlit

