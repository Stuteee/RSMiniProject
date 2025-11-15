from flask import Flask, request, jsonify, render_template
import os
import pandas as pd

from hybrid import HybridRecommender
import random

app = Flask(__name__)

# Base and model directories
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'recommender_top5k')

# instantiate recommender
RECS = HybridRecommender(MODEL_DIR, base_dir=BASE_DIR)


@app.route('/')
def index():
    # support optional search query via ?q=...
    q = request.args.get('q')
    if not q:
        # get a larger pool of popular books and randomly sample a subset each load
        pool = RECS.get_top_books(top_n=100)
        pool_titles = pool['titles']
        pool_images = pool['images']
        pool_authors = pool['authors']
        pool_votes = pool['votes']
        pool_rating = pool['rating']

        DISPLAY_COUNT = 20
        if len(pool_titles) > DISPLAY_COUNT:
            idxs = random.sample(range(len(pool_titles)), DISPLAY_COUNT)
        else:
            idxs = list(range(len(pool_titles)))

        titles = [pool_titles[i] for i in idxs]
        images = [pool_images[i] for i in idxs]
        authors = [pool_authors[i] for i in idxs]
        votes = [pool_votes[i] for i in idxs]
        rating = [pool_rating[i] for i in idxs]
        pool_book_ids = pool.get('book_ids', [])
        book_ids = [pool_book_ids[i] if i < len(pool_book_ids) else '' for i in idxs]

        return render_template('index.html', book_name=titles, image=images, author=authors, votes=votes, rating=rating, book_ids=book_ids, query='')

    # search in active books
    books = RECS.ART['books'].copy()
    books['title_str'] = books.get('title', '').astype(str)
    # substring match (case-insensitive)
    mask = books['title_str'].str.contains(q, case=False, na=False)
    results_df = books[mask]

    # if no substring matches, try fuzzy matching to nearest title
    if results_df.empty:
        best = RECS.fuzzy_match_title(q, list(books['title_str'].astype(str)))
        if best:
            results_df = books[books['title_str'] == best]

    # if still empty, try matching authors
    if results_df.empty and 'authors' in books.columns:
        authors = books['authors'].astype(str)
        mask2 = authors.str.contains(q, case=False, na=False)
        results_df = books[mask2]

    # Fallback: return top books if nothing found
    if results_df.empty:
        top = RECS.get_top_books(top_n=50)
        return render_template('index.html', book_name=top['titles'], image=top['images'], author=top['authors'], votes=top['votes'], rating=top['rating'], book_ids=top.get('book_ids', []), query=q, msg=f'No exact matches for "{q}", showing popular books.')

    # prepare lists for template
    results_df = results_df.copy()
    results_df['book_id'] = results_df['book_id'].astype(str)
    # votes and rating maps via Ratings.csv like HybridRecommender
    ratings_path = os.path.join(BASE_DIR, 'Ratings.csv')
    votes_map = {}
    rating_map = {}
    if os.path.exists(ratings_path):
        try:
            ratings_df = pd.read_csv(ratings_path, encoding='utf-8')
            if 'Book-Rating' in ratings_df.columns and 'ISBN' in ratings_df.columns:
                ratings_df = ratings_df.rename(columns={'Book-Rating': 'rating', 'ISBN': 'book_id', 'User-ID': 'user_id'})
            if {'book_id', 'rating'}.issubset(ratings_df.columns):
                agg = ratings_df.groupby('book_id').agg({'rating': ['count', 'mean']}).reset_index()
                agg.columns = ['book_id', 'rating_count', 'rating_mean']
                votes_map = dict(zip(agg['book_id'].astype(str), agg['rating_count']))
                rating_map = dict(zip(agg['book_id'].astype(str), agg['rating_mean']))
        except Exception:
            votes_map = {}
            rating_map = {}

    if 'image' in results_df.columns:
        images = list(results_df['image'].fillna(''))
    elif 'Image-URL-L' in results_df.columns:
        images = list(results_df['Image-URL-L'].fillna(''))
    else:
        images = [''] * len(results_df)

    titles = list(results_df.get('title', results_df.get('book_id')).astype(str))
    authors = list(results_df.get('authors', '').astype(str))
    votes = [int(votes_map.get(bid, 0)) for bid in results_df['book_id'].astype(str)]
    rating = [float(rating_map.get(bid, 0.0)) for bid in results_df['book_id'].astype(str)]
    book_ids = list(results_df['book_id'].astype(str))

    return render_template('index.html', book_name=titles, image=images, author=authors, votes=votes, rating=rating, book_ids=book_ids, query=q)


@app.route('/recommend_ui')
def recommend_ui():
    return render_template('recommend.html')


@app.route('/contact')
def contact():
    """Render the contact / about page describing the project."""
    return render_template('contact.html')


@app.route('/rate/<book_id>')
def rate_book(book_id):
    """Render the book rating page with recommendations after rating."""
    books = RECS.ART['books'].copy()
    books['book_id'] = books['book_id'].astype(str)
    
    # Find the book
    book_row = books[books['book_id'] == book_id]
    if book_row.empty:
        return "Book not found", 404
    
    book_row = book_row.iloc[0]
    
    # Get votes and rating from Ratings.csv
    ratings_path = os.path.join(BASE_DIR, 'Ratings.csv')
    votes = 0
    rating = 0.0
    if os.path.exists(ratings_path):
        try:
            ratings_df = pd.read_csv(ratings_path, encoding='utf-8')
            if 'Book-Rating' in ratings_df.columns and 'ISBN' in ratings_df.columns:
                ratings_df = ratings_df.rename(columns={'Book-Rating': 'rating', 'ISBN': 'book_id', 'User-ID': 'user_id'})
            if {'book_id', 'rating'}.issubset(ratings_df.columns):
                book_ratings = ratings_df[ratings_df['book_id'].astype(str) == book_id]
                if not book_ratings.empty:
                    votes = int(book_ratings['rating'].count())
                    rating = float(book_ratings['rating'].mean())
        except Exception:
            pass
    
    # Get image URL
    image = ''
    if 'image' in book_row:
        image = book_row.get('image', '')
    elif 'Image-URL-L' in book_row:
        image = book_row.get('Image-URL-L', '')
    
    book_data = {
        'book_id': str(book_id),
        'title': str(book_row.get('title', 'Unknown')),
        'authors': str(book_row.get('authors', 'Unknown')),
        'image': image,
        'votes': votes,
        'rating': rating if rating > 0 else 0.0
    }
    
    return render_template('rate.html', book=book_data)


@app.route('/recommend', methods=['GET'])
def recommend():
    q_title = request.args.get('title')
    q_book_id = request.args.get('book_id')
    n = int(request.args.get('n', 10))
    w_collab = float(request.args.get('w_collab', 0.6))
    w_content = float(request.args.get('w_content', 0.3))
    w_pop = float(request.args.get('w_pop', 0.1))
    try:
        resp = RECS.recommend(q_title=q_title, q_book_id=q_book_id, n=n, w_collab=w_collab, w_content=w_content, w_pop=w_pop)
        return jsonify(resp)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)