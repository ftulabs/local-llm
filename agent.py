#!/usr/bin/env python3
"""Lightweight tool-calling agent using local llama-server."""

import argparse
import json
import os
import sys
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling schema)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information. Use this when the user asks about recent events, facts you're unsure about, or anything that needs up-to-date data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "Search through ingested documents for relevant information. Use this when the user asks about content from uploaded/ingested files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 3)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_ingest",
            "description": "Ingest a text file into the knowledge base for later retrieval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the text file to ingest",
                    },
                    "chunk_size": {
                        "type": "integer",
                        "description": "Size of text chunks in characters (default: 500)",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo."""
    try:
        from ddgs import DDGS

        results = list(DDGS().text(query, max_results=max_results))

        if not results:
            return "No results found."

        output = []
        for i, r in enumerate(results, 1):
            output.append(f"[{i}] {r['title']}\n    {r['href']}\n    {r['body']}")
        return "\n\n".join(output)
    except Exception as e:
        return f"Search error: {e}"


def rag_search(query: str, n_results: int = 3) -> str:
    """Search the RAG knowledge base."""
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(RAG_DB_PATH))
        collection = client.get_or_create_collection("documents")

        if collection.count() == 0:
            return "Knowledge base is empty. Use rag_ingest to add documents first."

        results = collection.query(query_texts=[query], n_results=min(n_results, collection.count()))

        output = []
        for i, (doc, meta, dist) in enumerate(
            zip(results["documents"][0], results["metadatas"][0], results["distances"][0]), 1
        ):
            source = meta.get("source", "unknown")
            output.append(f"[{i}] (score: {1 - dist:.3f}) source: {source}\n    {doc}")
        return "\n\n".join(output) if output else "No relevant documents found."
    except Exception as e:
        return f"RAG search error: {e}"


def rag_ingest(file_path: str, chunk_size: int = 500) -> str:
    """Ingest a file into the RAG knowledge base."""
    try:
        import chromadb

        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            return f"File not found: {path}"

        text = path.read_text(encoding="utf-8", errors="replace")
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

        if not chunks:
            return "File is empty."

        client = chromadb.PersistentClient(path=str(RAG_DB_PATH))
        collection = client.get_or_create_collection("documents")

        ids = [f"{path.name}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": str(path), "chunk": i} for i in range(len(chunks))]

        collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)

        return f"Ingested {len(chunks)} chunks from {path.name} ({len(text)} chars total)."
    except Exception as e:
        return f"Ingest error: {e}"


# Tool dispatch
TOOL_FUNCTIONS = {
    "web_search": web_search,
    "rag_search": rag_search,
    "rag_ingest": rag_ingest,
}

# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

RAG_DB_PATH = Path(__file__).parent / "rag_db"


def run_agent(
    user_message: str,
    base_url: str = "http://localhost:8899/v1",
    system_prompt: str = "You are a helpful assistant with access to web search and a document knowledge base. Use tools when needed to provide accurate, up-to-date answers.",
    max_turns: int = 5,
    verbose: bool = False,
):
    """Run the agent loop: send message -> handle tool calls -> return final answer."""
    client = OpenAI(base_url=base_url, api_key="not-needed")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    for turn in range(max_turns):
        if verbose:
            print(f"\n--- Turn {turn + 1} ---", file=sys.stderr)

        response = client.chat.completions.create(
            model="qwen",
            messages=messages,
            tools=TOOLS,
            max_tokens=1024,
        )

        choice = response.choices[0]
        message = choice.message

        # If no tool calls, return the final text
        if choice.finish_reason != "tool_calls" or not message.tool_calls:
            return message.content or ""

        # Append assistant message with tool calls
        messages.append(message.model_dump())

        # Execute each tool call
        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            if verbose:
                print(f"  Tool: {fn_name}({fn_args})", file=sys.stderr)

            fn = TOOL_FUNCTIONS.get(fn_name)
            if fn:
                result = fn(**fn_args)
            else:
                result = f"Unknown tool: {fn_name}"

            if verbose:
                preview = result[:200] + "..." if len(result) > 200 else result
                print(f"  Result: {preview}", file=sys.stderr)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

    return "Max tool call turns reached."


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def interactive_mode(base_url: str, verbose: bool):
    """Run interactive chat loop."""
    print("Agent ready. Type your message (Ctrl+C to exit).")
    print("Commands: /ingest <file>  — add file to knowledge base")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue

        # Shortcut for ingesting files
        if user_input.startswith("/ingest "):
            file_path = user_input[8:].strip()
            result = rag_ingest(file_path)
            print(f"Agent: {result}\n")
            continue

        answer = run_agent(user_input, base_url=base_url, verbose=verbose)
        print(f"Agent: {answer}\n")


def main():
    parser = argparse.ArgumentParser(description="Local LLM agent with tools")
    parser.add_argument("message", nargs="*", help="Message to send (interactive mode if omitted)")
    parser.add_argument("--url", default="http://localhost:8899/v1", help="LLM server URL")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show tool calls")
    args = parser.parse_args()

    if args.message:
        answer = run_agent(" ".join(args.message), base_url=args.url, verbose=args.verbose)
        print(answer)
    else:
        interactive_mode(args.url, args.verbose)


if __name__ == "__main__":
    main()
