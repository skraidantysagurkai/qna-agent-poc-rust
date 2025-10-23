import argparse
import json
import os
from urllib.parse import urlparse
from typing import List

import re
from pydantic import BaseModel

from openai import AsyncClient
from openai import Embedding

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, CollectionStatus
import asyncio

import logging


def get_logger(name: str):
    """Set up and return a logger with a standard format."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate log entries in case of multiple imports
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = get_logger(__name__)


class TextEntry(BaseModel):
    url: str
    content: str


class ContextEntry(BaseModel):
    content: TextEntry
    vector: List[float]


class TextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separators: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    def split_text(self, text: str, metadata: dict = None) -> List[dict]:
        if not text:
            return []

        text_chunks = self._split_text_recursive(text, self.separators)

        chunk_objects = []
        for chunk_index, chunk_text in enumerate(text_chunks):
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update(
                {"chunk_index": chunk_index, "total_chunks": len(text_chunks), "chunk_size": len(chunk_text)}
            )

            chunk_objects.append({"content": chunk_text.strip(), "metadata": chunk_metadata})

        return chunk_objects

    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        if not separators:
            return [text]

        current_separator = separators[0]
        remaining_separators = separators[1:]

        if current_separator == "":
            return self._split_by_characters(text)

        text_splits = text.split(current_separator)
        text_chunks = []
        current_chunk = ""

        for text_split in text_splits:
            potential_chunk_size = len(current_chunk) + len(text_split) + len(current_separator)

            if potential_chunk_size <= self.chunk_size:
                if current_chunk:
                    current_chunk += current_separator + text_split
                else:
                    current_chunk = text_split
            else:
                if current_chunk:
                    text_chunks.append(current_chunk)

                if len(text_split) > self.chunk_size:
                    sub_chunks = self._split_text_recursive(text_split, remaining_separators)
                    text_chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = text_split

        if current_chunk:
            text_chunks.append(current_chunk)

        return self._merge_small_chunks(text_chunks)

    def _split_by_characters(self, text: str) -> List[str]:
        character_chunks = []
        for start_index in range(0, len(text), self.chunk_size - self.chunk_overlap):
            character_chunk = text[start_index : start_index + self.chunk_size]
            character_chunks.append(character_chunk)
        return character_chunks

    def _merge_small_chunks(self, text_chunks: List[str]) -> List[str]:
        if not text_chunks:
            return text_chunks

        merged_chunks = []
        current_merged_chunk = text_chunks[0]

        for next_chunk in text_chunks[1:]:
            combined_size = len(current_merged_chunk) + len(next_chunk) + 1

            if combined_size <= self.chunk_size:
                current_merged_chunk += " " + next_chunk
            else:
                merged_chunks.append(current_merged_chunk)

                if self.chunk_overlap > 0 and len(current_merged_chunk) > self.chunk_overlap:
                    overlap_text = current_merged_chunk[-self.chunk_overlap :]
                    current_merged_chunk = overlap_text + " " + next_chunk
                else:
                    current_merged_chunk = next_chunk

        merged_chunks.append(current_merged_chunk)
        return merged_chunks


class RawDataPreprocessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    @staticmethod
    def read_json_file(file_path: str) -> List[TextEntry]:
        try:
            with open(file_path, "r", encoding="utf-8") as json_file:
                json_data = json.load(json_file)
                if isinstance(json_data, dict):
                    json_data = [json_data]
                return [
                    TextEntry(**data_item) for data_item in json_data if "url" in data_item and "content" in data_item
                ]
        except Exception as error:
            logger.error(f"Error reading JSON file {file_path}: {error}")
            raise

    @staticmethod
    def extract_title_from_url(url: str) -> str:
        try:
            parsed_url = urlparse(url)
            url_path_segments = [segment for segment in parsed_url.path.strip("//").split("/") if segment]

            if not url_path_segments:
                return ""

            title_segments = url_path_segments[1:]

            raw_title = " ".join(title_segments)
            title_without_separators = re.sub(r"[-_]", " ", raw_title)
            title_alphanumeric_only = re.sub(r"[^a-zA-Z0-9\s]", "", title_without_separators)
            formatted_title = " ".join(word.capitalize() for word in title_alphanumeric_only.split())

            return formatted_title if formatted_title else ""

        except Exception as error:
            logger.warning(f"Error extracting title from URL {url}: {error}")
            return "Web Content"

    @staticmethod
    def clean_text_content(content: str) -> str:
        if not content:
            return ""

        content_url_safe = re.sub(r"[^a-zA-Z0-9.,!?;:\-_/()\"'&%#=~\n ]+", "", content)
        content_single_newlines = re.sub(r"\n+", "\n", content_url_safe)
        content_single_spaces = re.sub(r"[^\S\n]+", " ", content_single_newlines)
        content_punctuation_fixed = re.sub(r"\s+([.,!?;:])", r"\1", content_single_spaces)
        content_duplicate_punctuation_removed = re.sub(r"([.,!?;:])\1+", r"\1", content_punctuation_fixed)
        return content_duplicate_punctuation_removed.strip()

    def process_text_entries(self, text_entries: List[TextEntry]) -> List[ContextEntry]:
        processed_context_entries = []

        for text_entry in text_entries:
            entry_url = text_entry.url
            entry_content = text_entry.content

            url_title = self.extract_title_from_url(entry_url)
            cleaned_content = self.clean_text_content(entry_content)

            if cleaned_content:
                formatted_content = f"<{url_title}>\n{cleaned_content}"

                chunk_metadata = {"source_url": entry_url, "title": url_title, "original_length": len(cleaned_content)}

                content_chunks = self.text_splitter.split_text(formatted_content, chunk_metadata)

                for chunk_data in content_chunks:
                    chunk_content = chunk_data["content"]
                    chunk_metadata = chunk_data["metadata"]

                    chunk_url = f"{entry_url}#chunk_{chunk_metadata['chunk_index']}"
                    chunk_text_entry = TextEntry(url=chunk_url, content=chunk_content)

                    processed_context_entries.append(ContextEntry(content=chunk_text_entry, vector=[]))
            else:
                logger.warning(f"Empty content after cleaning for URL: {entry_url}")

        logger.info(f"Created {len(processed_context_entries)} chunks from {len(text_entries)} original entries")
        return processed_context_entries

    @staticmethod
    def _remove_duplicate_entries(text_entries: List[TextEntry]) -> List[TextEntry]:
        return list(set(text_entries))

    def process_json_file(self, file_path: str) -> List[ContextEntry]:
        logger.info(f"Processing JSON file: {file_path}")

        raw_text_entries = self.read_json_file(file_path)
        processed_context_entries = self.process_text_entries(raw_text_entries)

        logger.info(f"Processed {len(processed_context_entries)} URL-content pairs")
        return processed_context_entries


class EmbeddingGenerator:
    def __init__(self, api_key: str):
        self.openai_client = AsyncClient(api_key=api_key)
        self.embedding_model_name = "text-embedding-3-small"

        self.max_tokens_per_entry = 8191
        self.max_entries_per_batch = 2048
        self.max_tokens_per_batch = 300000

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def embed_text(self, text_content: str) -> Embedding:
        try:
            embedding_response = await self.openai_client.embeddings.create(
                model=self.embedding_model_name, input=text_content
            )
            return embedding_response.data[0].embedding
        except Exception as error:
            logger.warning(f"Embedding attempt failed: {error}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _embed_text_batch(self, context_entries: List[ContextEntry]) -> List[ContextEntry]:
        try:
            batch_texts = [context_entry.content.content for context_entry in context_entries]
            batch_response = await self.openai_client.embeddings.create(
                model=self.embedding_model_name, input=batch_texts
            )
            return [
                ContextEntry(content=context_entry.content, vector=embedding_data.embedding)
                for context_entry, embedding_data in zip(context_entries, batch_response.data)
            ]
        except Exception as error:
            logger.warning(f"Embedding attempt failed: {error}")
            raise

    async def embed_text_in_batches(self, context_entries: List[ContextEntry]) -> List[ContextEntry]:
        all_embedded_entries = []
        entry_batches = self.batch_texts(context_entries)

        for entry_batch in entry_batches:
            batch_embeddings = await self._embed_text_batch(entry_batch)
            all_embedded_entries.extend(batch_embeddings)

        return all_embedded_entries

    def batch_texts(self, context_entries: List[ContextEntry]) -> List[List[ContextEntry]]:
        entry_batches = []
        current_batch = []
        current_batch_token_count = 0

        for context_entry in context_entries:
            entry_text = context_entry.content.content
            entry_token_count = len(entry_text.encode("utf-8")) * 0.25

            if entry_token_count > self.max_tokens_per_entry:
                logger.warning(f"Text entry exceeds max tokens ({self.max_tokens_per_entry}), skipping.")
                continue

            batch_would_exceed_limits = (
                len(current_batch) + 1 > self.max_entries_per_batch
                or current_batch_token_count + entry_token_count > self.max_tokens_per_batch
            )

            if batch_would_exceed_limits:
                entry_batches.append(current_batch)
                current_batch = [context_entry]
                current_batch_token_count = entry_token_count
            else:
                current_batch.append(context_entry)
                current_batch_token_count += entry_token_count

        if current_batch:
            entry_batches.append(current_batch)

        return entry_batches


class DatabaseInterface:
    def __init__(self, database_url: str):
        self.qdrant_client = QdrantClient(url=database_url)

    def _ensure_collection_exists(self, collection_name: str, vector_size: int = 1536) -> None:
        try:
            collection_info = self.qdrant_client.get_collection(collection_name)
            if collection_info.status == CollectionStatus.GREEN:
                logger.info(f"Collection '{collection_name}' already exists and is ready")
                return
        except Exception:
            logger.info(f"Collection '{collection_name}' does not exist, creating it")

        try:
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info(f"Successfully created collection '{collection_name}'")
        except Exception as error:
            logger.error(f"Failed to create collection '{collection_name}': {error}")
            raise

    def populate_database(self, context_entries: List[ContextEntry], collection_name: str) -> None:
        if not context_entries:
            logger.warning("No context entries to populate")
            return

        vector_size = len(context_entries[0].vector) if context_entries[0].vector else 1536
        self._ensure_collection_exists(collection_name, vector_size)

        entry_vectors = [context_entry.vector for context_entry in context_entries]
        entry_payloads = [context_entry.content.model_dump() for context_entry in context_entries]

        self.qdrant_client.upload_collection(
            collection_name=collection_name,
            payload=entry_payloads,
            vectors=entry_vectors,
            parallel=4,
            max_retries=3,
        )

        logger.info(f"Successfully populated collection '{collection_name}' with {len(context_entries)} entries")


async def main(database_url: str, collection_name: str, input_file_path: str, openai_api_key: str) -> None:
    data_preprocessor = RawDataPreprocessor()
    database_interface = DatabaseInterface(database_url=database_url)
    embedding_generator = EmbeddingGenerator(api_key=openai_api_key)

    processed_context_entries = data_preprocessor.process_json_file(input_file_path)
    embedded_context_entries = await embedding_generator.embed_text_in_batches(processed_context_entries)
    database_interface.populate_database(embedded_context_entries, collection_name=collection_name)


def build_arg_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description="Read robot.txt from a website.")
    argument_parser.add_argument(
        "-u",
        "--db_url",
        type=str,
        required=True,
        help="Full url with port of the database to populate",
    )
    argument_parser.add_argument(
        "-i",
        "--input_file_path",
        type=str,
        required=True,
        help="Input JSON file name with which to populate the database",
    )
    argument_parser.add_argument(
        "-c",
        "--db_collection_name",
        type=str,
        required=True,
        help="Collection name where to insert the data",
    )
    argument_parser.add_argument(
        "--openai_api_key",
        type=str,
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAi API Key, if not set, taken from environment",
    )

    return argument_parser


if __name__ == "__main__":
    command_line_parser = build_arg_parser()
    parsed_arguments = command_line_parser.parse_args()

    asyncio.run(
        main(
            parsed_arguments.db_url,
            parsed_arguments.db_collection_name,
            parsed_arguments.input_file_path,
            parsed_arguments.openai_api_key,
        )
    )
