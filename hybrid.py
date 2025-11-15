import os
import pickle
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import difflib


class HybridRecommender:
    """Lightweight hybrid recommender utility.

    Methods:
    - load_artifacts(): load persisted TFIDF, SVD and mappings from a model folder
    - recommend(...): compute content / collaborative / popularity scores and return top-n results
    - get_top_books(): helper to return top books for index page
    """

    def __init__(self, model_dir, base_dir=None):
        self.model_dir = model_dir
        self.base_dir = base_dir or os.path.dirname(__file__)
        self.ART = self.load_artifacts()
        self.POP_MAP = self.compute_popularity_scores(self.ART['books'])

    def load_artifacts(self):
        model_dir = self.model_dir
        artifacts = {}
        artifacts['books'] = pd.read_pickle(os.path.join(model_dir, 'active_books.pkl'))
        with open(os.path.join(model_dir, 'tfidf.pkl'), 'rb') as f:
            artifacts['tfidf'] = pickle.load(f)
        artifacts['tfidf_matrix'] = sparse.load_npz(os.path.join(model_dir, 'tfidf_matrix.npz'))

        mappings_path = os.path.join(model_dir, 'mappings.pkl')
        if os.path.exists(mappings_path):
            with open(mappings_path, 'rb') as f:
                maps = pickle.load(f)
            artifacts['item_index_to_id'] = maps.get('item_index_to_id', [])
            artifacts['user_index_to_id'] = maps.get('user_index_to_id', [])
        else:
            artifacts['item_index_to_id'] = []
            artifacts['user_index_to_id'] = []

        svd_vt_path = os.path.join(model_dir, 'svd_vt.npy')
        if os.path.exists(svd_vt_path):
            artifacts['svd_vt'] = np.load(svd_vt_path)
            artifacts['svd_u'] = np.load(os.path.join(model_dir, 'svd_u.npy'))
            artifacts['svd_sigma'] = np.load(os.path.join(model_dir, 'svd_sigma.npy'))
            artifacts['collaborative_ready'] = True
            artifacts['item_vectors'] = artifacts['svd_vt'].T
        else:
            artifacts['svd_vt'] = None
            artifacts['svd_u'] = None
            artifacts['svd_sigma'] = None
            artifacts['item_vectors'] = None
            artifacts['collaborative_ready'] = False

        return artifacts

    @staticmethod
    def normalize_scores(arr):
        arr = np.array(arr, dtype=float)
        if arr.size == 0:
            return arr
        mn = np.nanmin(arr)
        mx = np.nanmax(arr)
        if mx - mn == 0:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    def compute_popularity_scores(self, books_df):
        ratings_path = os.path.join(self.base_dir, 'Ratings.csv')
        if not os.path.exists(ratings_path):
            return {}
        ratings = pd.read_csv(ratings_path, encoding='utf-8')
        if 'Book-Rating' in ratings.columns and 'ISBN' in ratings.columns:
            ratings = ratings.rename(columns={'Book-Rating': 'rating', 'ISBN': 'book_id', 'User-ID': 'user_id'})
        if 'book_id' in ratings.columns:
            ratings['book_id'] = ratings['book_id'].astype(str)
        if 'rating' not in ratings.columns:
            return {}

        agg = ratings.groupby('book_id').agg({'rating': ['count', 'mean']})
        agg.columns = ['rating_count', 'rating_mean']
        agg = agg.reset_index()
        scaler = MinMaxScaler()
        agg[['rating_count_norm', 'rating_mean_norm']] = scaler.fit_transform(agg[['rating_count', 'rating_mean']].fillna(0))
        agg['popularity'] = agg['rating_count_norm'] * 0.7 + agg['rating_mean_norm'] * 0.3
        return dict(zip(agg['book_id'].astype(str), agg['popularity']))

    @staticmethod
    def fuzzy_match_title(title, titles_list, cutoff=0.6):
        matches = difflib.get_close_matches(title, titles_list, n=1, cutoff=cutoff)
        if matches:
            return matches[0]
        q = title.lower()
        for t in titles_list:
            if q in t.lower():
                return t
        return None

    def recommend(self, q_title=None, q_book_id=None, n=10, w_collab=0.6, w_content=0.3, w_pop=0.1):
        books = self.ART['books']

        # find book index and id
        target_idx = None
        target_book_id = None
        if q_title:
            cand = books[books.get('title', '') == q_title]
            if cand.empty:
                best = self.fuzzy_match_title(q_title, list(books['title'].astype(str)))
                if best is None:
                    raise ValueError('title not found')
                cand = books[books.get('title', '') == best]
            if cand.empty:
                raise ValueError('title lookup failed')
            target_idx = int(cand.index[0])
            target_book_id = str(cand.iloc[0].get('book_id'))
        elif q_book_id:
            cand = books[books.get('book_id', '') == str(q_book_id)]
            if cand.empty:
                raise ValueError('book_id not found')
            target_idx = int(cand.index[0])
            target_book_id = str(q_book_id)
        else:
            raise ValueError('provide title or book_id')

        # content
        if self.ART.get('tfidf_matrix') is None:
            content_scores = np.zeros(len(books))
        else:
            vec = self.ART['tfidf_matrix'][target_idx]
            content_scores = cosine_similarity(vec, self.ART['tfidf_matrix']).ravel()
        content_scores = self.normalize_scores(content_scores)

        # collaborative
        if self.ART.get('collaborative_ready') and self.ART.get('item_vectors') is not None and self.ART.get('item_index_to_id'):
            try:
                item_idx = self.ART['item_index_to_id'].index(target_book_id)
                item_vecs = self.ART['item_vectors']
                collab_raw = item_vecs.dot(item_vecs[item_idx])
                id_to_active_pos = {str(bid): pos for pos, bid in enumerate(books['book_id'].astype(str))}
                collab_scores = np.zeros(len(books))
                for idx_i, bid in enumerate(self.ART['item_index_to_id']):
                    pos = id_to_active_pos.get(str(bid))
                    if pos is not None:
                        collab_scores[pos] = collab_raw[idx_i]
            except ValueError:
                collab_scores = np.zeros(len(books))
        else:
            collab_scores = np.zeros(len(books))
        collab_scores = self.normalize_scores(collab_scores)

        # popularity
        pop_scores = np.array([self.POP_MAP.get(str(bid), 0.0) for bid in books['book_id'].astype(str)])
        pop_scores = self.normalize_scores(pop_scores)

        hybrid_raw = w_collab * collab_scores + w_content * content_scores + w_pop * pop_scores
        hybrid_scores = self.normalize_scores(hybrid_raw)

        # augment and build df
        books_local = books.copy()
        ratings_path = os.path.join(self.base_dir, 'Ratings.csv')
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

        img_col = None
        for c in ['image', 'Image-URL-L', 'image_url', 'cover', 'Image-URL-M', 'Image-URL-S']:
            if c in books_local.columns:
                img_col = c
                break

        df = books_local[['title', 'authors', 'book_id']].copy()
        df['image'] = books_local[img_col].fillna('') if img_col else ''
        df['votes'] = books_local['book_id'].astype(str).map(votes_map).fillna(0).astype(int)
        df['rating_avg'] = books_local['book_id'].astype(str).map(rating_map).fillna(0.0)
        df['content_score'] = content_scores
        df['collab_score'] = collab_scores
        df['pop_score'] = pop_scores
        df['hybrid_score'] = hybrid_scores

        df = df.reset_index(drop=True)
        if target_book_id is not None:
            df = df[df['book_id'].astype(str) != str(target_book_id)]

        df_sorted = df.sort_values('hybrid_score', ascending=False)
        results = []
        seen_authors = set()
        for _, row in df_sorted.iterrows():
            author = str(row.get('authors', '')).lower()
            score = float(row['hybrid_score'])
            if author in seen_authors:
                score *= 0.9
            else:
                seen_authors.add(author)
            results.append({
                'title': row['title'],
                'authors': row['authors'],
                'book_id': row['book_id'],
                'image': row.get('image', ''),
                'votes': int(row.get('votes', 0)),
                'rating': float(row.get('rating_avg', 0.0)),
                'content_score': float(row['content_score']),
                'collab_score': float(row['collab_score']),
                'pop_score': float(row['pop_score']),
                'hybrid_score': score
            })
            if len(results) >= n:
                break

        return {'query': q_title or q_book_id, 'n': n, 'results': results}

    def get_top_books(self, top_n=50):
        books = self.ART['books'].copy()
        ratings_path = os.path.join(self.base_dir, 'Ratings.csv')
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

        books['book_id'] = books['book_id'].astype(str)
        books['votes'] = books['book_id'].map(votes_map).fillna(0).astype(int)
        books['rating_avg'] = books['book_id'].map(rating_map).fillna(0.0)
        top = books.sort_values('votes', ascending=False).head(top_n)

        # images
        if 'image' in top.columns:
            images = list(top['image'].fillna(''))
        elif 'Image-URL-L' in top.columns:
            images = list(top['Image-URL-L'].fillna(''))
        else:
            images = [''] * len(top)

        return {
            'titles': list(top.get('title', top.get('book_id')).astype(str)),
            'authors': list(top.get('authors', '').astype(str)),
            'images': images,
            'votes': list(top['votes']),
            'rating': list(top['rating_avg']),
            'book_ids': list(top['book_id'].astype(str))
        }
