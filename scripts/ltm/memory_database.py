"""Long Term Memory database for MonikA.I.

Stores and retrieves conversation memories using semantic embeddings.
Uses SQLite for text storage and Zarr for embedding vector storage.
Retrieval is based on cosine similarity via SentenceTransformer embeddings.
"""

import pathlib
import sqlite3

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import zarr

from scripts.ltm.constants import (
    CHUNK_SIZE,
    DATABASE_NAME,
    EMBEDDINGS_NAME,
    EMBEDDING_VECTOR_LENGTH,
    SENTENCE_TRANSFORMER_MODEL,
)
from scripts.ltm.queries import (
    CREATE_TABLE_QUERY,
    DROP_TABLE_QUERY,
    FETCH_DATA_QUERY,
    INSERT_DATA_QUERY,
)


class LtmDatabase:
    """API over a Long Term Memory database."""

    def __init__(
        self,
        directory,
        num_memories_to_fetch=1,
        force_use_legacy_db=False,
    ):
        """Loads all resources."""
        self.directory = pathlib.Path(directory)

        self.database_path = None
        self.embeddings_path = None

        self.character_name = None
        self.message_embeddings = None
        self.disk_embeddings = None
        self.sql_conn = None

        # Load db
        (legacy_database_path, legacy_embeddings_path) = self._build_database_paths()
        legacy_db_exists = legacy_database_path.exists() and legacy_embeddings_path.exists()
        use_legacy_db = force_use_legacy_db or legacy_db_exists
        if use_legacy_db:
            print("=" * 20)
            print("WARNING: LEGACY DATABASE DETECTED, CHARACTER NAMESPACING IS DISABLED")
            print("         Memories will be shared across all characters.")
            print("=" * 20)
            self.database_path = legacy_database_path
            self.embeddings_path = legacy_embeddings_path
            self._load_db()

        # Load analytic modules
        print("Loading SentenceTransformer model for Long Term Memory...")
        self.sentence_embedder = SentenceTransformer(
            SENTENCE_TRANSFORMER_MODEL, device="cpu"
        )
        self.num_memories_to_fetch = num_memories_to_fetch
        print("SentenceTransformer model loaded successfully.")

        # Set legacy status
        self.use_legacy_db = use_legacy_db

    def _build_database_paths(self, character_name=None):
        if character_name is None:
            database_path = self.directory / DATABASE_NAME
            embeddings_path = self.directory / EMBEDDINGS_NAME
        else:
            database_path = self.directory / character_name / DATABASE_NAME
            embeddings_path = self.directory / character_name / EMBEDDINGS_NAME

        return (database_path, embeddings_path)

    def _load_db(self, database_namespace="LEGACY_UNIFIED_DATABASE"):
        if not self.database_path.exists() and not self.embeddings_path.exists():
            print("No existing memories found for {}, "
                  "will create a new database.".format(database_namespace))
            self._destroy_and_recreate_database(do_sql_drop=False)
        elif self.database_path.exists() and not self.embeddings_path.exists():
            raise RuntimeError(
                "ERROR: Inconsistent state detected for {}: "
                "{} exists but {} does not. "
                "Her memories are likely safe, but you'll have to regen the "
                "embedding vectors yourself manually.".format(
                    database_namespace, self.database_path, self.embeddings_path
                )
            )
        elif not self.database_path.exists() and self.embeddings_path.exists():
            raise RuntimeError(
                "ERROR: Inconsistent state detected for {}: "
                "{} exists but {} does not. "
                "Please look for {} in another directory, "
                "if you can't find it, her memories may be lost.".format(
                    database_namespace, self.embeddings_path,
                    self.database_path, DATABASE_NAME
                )
            )

        # Prepare the memory database for retrieval
        # Load the embeddings to a local numpy array
        self.message_embeddings = zarr.open(self.embeddings_path, mode="r")[:]
        # Prepare a "connection" to the embeddings, but to store new LTMs on disk
        self.disk_embeddings = zarr.open(self.embeddings_path, mode="a")
        # Prepare a "connection" to the master database
        self.sql_conn = sqlite3.connect(self.database_path, check_same_thread=False)

    def _destroy_and_recreate_database(self, do_sql_drop=False):
        """Destroys and re-creates a new LTM database.

        WARNING: THIS WILL DESTROY ANY EXISTING LONG TERM MEMORY DATABASE.
                 DO NOT CALL THIS METHOD YOURSELF UNLESS YOU KNOW EXACTLY
                 WHAT YOU'RE DOING!
        """
        # Create directories if they don't exist
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        # Create new sqlite table to store the textual memories
        sql_conn = sqlite3.connect(self.database_path)
        with sql_conn:
            if do_sql_drop:
                sql_conn.execute(DROP_TABLE_QUERY)
            sql_conn.execute(CREATE_TABLE_QUERY)

        # Create new embeddings db to store the fuzzy keys for the
        # corresponding memory text.
        # WARNING: will destroy any existing embeddings db
        zarr.open(
            self.embeddings_path,
            mode="w",
            shape=(0, EMBEDDING_VECTOR_LENGTH),
            chunks=(CHUNK_SIZE, EMBEDDING_VECTOR_LENGTH),
            dtype="float32",
        )

    def load_character_db_if_new(self, character_name):
        """Loads the database associated with the specified character."""
        if self.use_legacy_db:
            # Using legacy database, do nothing
            return
        if self.character_name == character_name:
            # No change in character, do nothing.
            return

        print("Loading LTM database for character: {}".format(character_name))

        # Load db of new character.
        (self.database_path, self.embeddings_path) = self._build_database_paths(character_name)
        self._load_db(character_name)
        self.character_name = character_name

    def add(self, name, new_message):
        """Adds a single new sentence to the LTM database."""
        # Create the message embedding
        new_message_embedding = self.sentence_embedder.encode(new_message)
        new_message_embedding = np.expand_dims(new_message_embedding, axis=0)

        # This line is a bit tricky:
        # The embedding_index is the INDEX of the disk_embeddings' NEXT vector,
        # which happens to be the same as the current number of vectors.
        embedding_index = self.disk_embeddings.shape[0]

        # Add the message to the master database if not a dupe
        with self.sql_conn as cursor:
            try:
                cursor.execute(INSERT_DATA_QUERY, (embedding_index, name, new_message))
            except sqlite3.IntegrityError as err:
                if "UNIQUE constraint failed:" in str(err):
                    # We are trying to add a duplicate message. Just don't add
                    # anything and continue on as normal
                    print("---duplicate memory detected, not adding again---")
                    return

                # We encountered an unexpected error, raise as normal
                raise

            # Save memory to persistent storage, if not a dupe
            self.disk_embeddings.append(new_message_embedding)

    def query(self, query_text):
        """Queries for the most similar sentences from the LTM database."""
        # If too few LTM features are loaded, return nothing.
        if self.message_embeddings is None or self.message_embeddings.shape[0] == 0:
            return []

        # Create the query embedding
        query_text_embedding = self.sentence_embedder.encode(query_text)
        query_text_embedding = np.expand_dims(query_text_embedding, axis=0)

        # Find the most relevant memory's index
        embedding_searcher = NearestNeighbors(
            n_neighbors=min(self.num_memories_to_fetch, self.message_embeddings.shape[0]),
            algorithm="brute",
            metric="cosine",
            n_jobs=-1,
        )
        embedding_searcher.fit(self.message_embeddings)
        (match_scores, embedding_indices) = embedding_searcher.kneighbors(
            query_text_embedding
        )

        all_query_responses = []
        for (match_score, embedding_index) in zip(match_scores[0], embedding_indices[0]):
            with self.sql_conn as cursor:
                response = cursor.execute(FETCH_DATA_QUERY, (int(embedding_index),))
                result = response.fetchone()
                if result is None:
                    continue
                (name, message, timestamp) = result

            query_response = {
                "name": name,
                "message": message,
                "timestamp": timestamp,
            }
            all_query_responses.append((query_response, match_score))

        return all_query_responses

    def reload_embeddings_from_disk(self):
        """Reloads all embeddings from disk into memory."""
        if self.message_embeddings is None:
            return "No memory database loaded."

        print("--------------------------------")
        print("Loading all embeddings from disk")
        print("--------------------------------")
        num_prior_embeddings = self.message_embeddings.shape[0]
        self.message_embeddings = zarr.open(self.embeddings_path, mode="r")[:]
        num_curr_embeddings = self.message_embeddings.shape[0]
        print("DONE!")
        print("Before: {} embeddings in memory".format(num_prior_embeddings))
        print("After: {} embeddings in memory".format(num_curr_embeddings))
        print("--------------------------------")
        return "Reloaded: {} -> {} memories".format(num_prior_embeddings, num_curr_embeddings)

    def destroy_all_memories(self):
        """Deletes all embeddings from memory AND disk."""
        if self.message_embeddings is None or self.disk_embeddings is None:
            return "No memory database loaded to destroy."

        print("--------------------------------------------------")
        print("Destroying all memories, I hope you backed them up")
        print("--------------------------------------------------")
        self.message_embeddings = None
        self.disk_embeddings = None

        self._destroy_and_recreate_database(do_sql_drop=True)

        self.disk_embeddings = zarr.open(self.embeddings_path, mode="a")
        self.message_embeddings = zarr.open(self.embeddings_path, mode="r")[:]
        print("DONE!")
        print("--------------------------------------------------")
        return "All memories have been destroyed."

    def get_stats(self):
        """Returns a dictionary of current LTM statistics."""
        num_memories_in_ram = (
            self.message_embeddings.shape[0]
            if self.message_embeddings is not None
            else 0
        )
        num_memories_on_disk = (
            self.disk_embeddings.shape[0]
            if self.disk_embeddings is not None
            else 0
        )
        return {
            "num_memories_in_ram": num_memories_in_ram,
            "num_memories_on_disk": num_memories_on_disk,
            "character": self.character_name or "legacy/default",
        }
