import streamlit as st
import pickle
from fuzzywuzzy import fuzz

st.header('Book Recommender System',)
#importing necassary files
books = pickle.load(open('dataset.pickle','rb'))
similarity = pickle.load(open('knn.pickle','rb'))
matrix = pickle.load(open('matrix.pickle','rb'))

books_list = books['bookTitle'].unique()
selected_book = st.selectbox("Type or select a movie from the dropdown",books_list)

# def recommend(book):
#     index = books[books['bookTitle'] == book].index[0]
#     distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
#     recommended_books_names = []
#     for i in distances[1:6]:
#         book_id = books.iloc[i[0]].book_id
#         recommended_books_names.append(books.iloc[i[0]].title)
#
#     return recommended_books_names
def print_book_recommendations(query_book):
    """
    Inputs:
    query_book: query artist name
    book_matrix: artist play count dataframe (not the sparse one, the pandas dataframe)
    knn_model: our previously fitted sklearn knn model
    k: the number of nearest neighbors.

    Prints: book recommendations for the query book
    Returns: None
    """
    rating_matrix=matrix
    knn_model= similarity
    k=10
    query_index = None
    ratio_tuples = []

    for i in rating_matrix.index:
        ratio = fuzz.ratio(i.lower(), query_book.lower())
        if ratio >= 75:
            current_query_index = rating_matrix.index.tolist().index(i)
            ratio_tuples.append((i, ratio, current_query_index))

    #st.text('Possible matches: {0}\n'.format([(x[0], x[1]) for x in ratio_tuples]))

    try:
        query_index = max(ratio_tuples, key=lambda x: x[1])[2]  # get the index of the best artist match in the data
    except:
        st.text('Your artist didn\'t match any artists in the data. Try again')
        return None

    distances, indices = knn_model.kneighbors(rating_matrix.iloc[query_index, :].values.reshape(1, -1),
                                              n_neighbors=k + 1)

    for i in range(0, len(distances.flatten())):
        if i == 0:
            st.text('Recommendations for {0}:\n'.format(rating_matrix.index[query_index]))
        else:
            st.text('{0}: {1}, with distance of {2}:'.format(i, rating_matrix.index[indices.flatten()[i]],
                                                           distances.flatten()[i]))

    return None

if st.button('Show Recommendation'):

    print_book_recommendations(selected_book)
    #st.write(print_book_recommendations(selected_book))