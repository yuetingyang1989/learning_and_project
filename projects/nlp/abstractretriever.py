from abc import abstractmethod, ABCMeta
import numpy as np
import logging
from itertools import islice
try:
    import faiss
except ImportError:
    faiss = None


logger = logging.getLogger('QueryRetriever')
logger.setLevel(logging.INFO)


def batch_reader(reader, batch_size):
    """Collect data into fixed-length chunks or blocks. The last block if not padded
    and is therefore not guaranteed to be of the same size as the previous blocks.

    Parameters
    ----------
    reader : iterable
        An type agnostic iterable (normally a stream) yielding a single item per iteration.
    batch_size : int
        Number of items to be wrapped in a batch.

    Returns
    -------
    iterable of tuple(T) where T is the type of `reader`
        Batched items from the original stream.
    """
    it = iter(reader)
    return iter(lambda: tuple(islice(it, batch_size)), ())


class QueryRetriever(metaclass=ABCMeta):
    def __init__(self):
        pass

    def score_tsv(self, filepath, batch_size=512):
        with open(filepath, "r") as f:
            distances = []
            for i, batch in enumerate(batch_reader(f.readlines(), batch_size=batch_size)):
                logger.info("Finished {} rows".format(i * batch_size))
                left = [line.strip().split("\t")[0] for line in batch]
                right = [line.strip().split("\t")[1] for line in batch]
                le = self.embed(left)
                re = self.embed(right)
                distance = np.linalg.norm(le - re, axis=1) ** 2
                distances.extend(distance)
            return distances

    def embed(self, data, batch_size=None, silent=True):
        if batch_size is None:
            return self._embed(data).numpy()

        start = True
        count = -1
        for batch in batch_reader(data, batch_size=batch_size):
            count +=1
            if not silent:
                logger.info("Finished {} samples".format(batch_size * count))
            embedded_batch = self._embed(batch).numpy()
            if start:
                embeddings = embedded_batch
                start = False
            else:
                embeddings = np.concatenate((embeddings, embedded_batch), axis=0)
        return embeddings

    def _rank(self, queries, db, k=None, batch_size=None, distance="euclidean"):
        if k is None:
            k = len(db)

        embedded_db = self.embed(db, batch_size=batch_size)
        embedded_queries = np.squeeze(
            np.asarray([self.embed([query]) for query in queries]),
            axis=1)

        if distance == "euclidean":
            index = faiss.IndexFlatL2(self.d)
        elif distance == "cosine":
            index = faiss.IndexFlatIP(self.d)
        else:
            raise ValueError("Unsupported distance {}, try one of 'euclidean', 'cosine'".format(distance))

        index.add(embedded_db)
        distances, neighbors = index.search(embedded_queries, k=k)
        return distances, neighbors

    def _compute_distances(self, queries, db, batch_size=None, ln=0.2, exclude=None):
        """Compute euclidean distances between all query and database pairs, potentially applying length normalization.

        Does not depend on FAISS.

        Parameters
        ----------
        queries : [str]
            All queries
        db : [str]
            All sentences to search against.
        batch_size : int, optional
            Batch to be used for embedding inference, only needed for very large inputs.
        ln : Length normalization.
            Inverts the logic of length penalty introduced in 'https://arxiv.org/abs/1609.08144'.

        Returns
        -------
        dict (int, int) -> float
            Mapping for pairs of query ID, sentence ID to their distance.
        """
        if ln == 0.0:
            length_norm = lambda x: 1
        else:
            length_norm = lambda s: ((5 + len(s)) ** ln) / (6 ** ln)

        embedded_db = self.embed(db, batch_size=batch_size)
        embedded_queries = np.asarray(
            [self._filter_embeddings[q] if q in self._filter_embeddings else self.embed(q) for q in queries]
        )
        if exclude is not None:
            embedded_queries -= self.embed(exclude)

        distances = {}
        for i in range(len(queries)):
            for j in range(len(db)):
                distances[(i, j)] = (np.linalg.norm(embedded_queries[i] - embedded_db[j]) ** 2) / length_norm(db[j])
        return distances

    @abstractmethod
    def rank_sentences(self, queries, sentences, k=5, distance="euclidean"):
        """Ranks the sentences according to their similarity to a given filter.

        Each context module (search filter) can be expressed through multiple queries, stored in `self.mapping`.
        When given a set of reviews, the distance score between that filter and the sentence, corresponds to the optimal
        (smallest) distance scored by any of the filter's queries.

        Two distances will be computed for the filter, one for its query. The final score will correspond to the distance
        between "Perfect breakfast" and "The hotel offered an amazing breakfast" because it will presumably be smaller
        than the other one.

        Parameters
        ----------
        queries : [str]
            The filters for which relevant sentences are identified.
        sentences : [str]
            Sentences to scan.
        k : int
            Number of top matches to return.
        distance : {'euclidean' | 'ip' | 'cosine'}
            Distance type to use for defining nearest neighbors.

        Returns
        -------
        ([int], [float])
            A tuple of two ordered lists. The first is the best review IDs and the second is their distances.
            The second list is therefore monotonically decreasing.
        """
        raise NotImplementedError("You must override this method.")

    @abstractmethod
    def rank_reviews(self, queries, reviews, k=5, distance="euclidean"):
        """If it was hard to write, it should be hard to read.

        Parameters
        ----------
        queries : [str]
            The filters to match against. Can span several queries.
        reviews : [str]
            List of reviews to be ranked, each of which can span multiple sentences.
        k : int
            Max number of reviews to return.
            We might return less if multiple sentences of the same review get matched. Sorry.
        distance : {'euclidean' | 'ip' | 'cosine'}
            Distance type to use for defining nearest neighbors.

        Returns
        -------
        [str] of shape n <= k
            n <=k reviews ordered by relevance to the passed search filter.
        """
        raise NotImplementedError("You must override this method.")