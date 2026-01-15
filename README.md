# Long-Term Memory

A LangChain-based agent system with persistent memory capabilities using PostgreSQL vector storage.

## Overview

This project implements an AI agent that can store and retrieve long-term memories across conversations. The agent uses semantic search to find relevant past interactions and personalizes responses based on stored context.

## Features

- Persistent memory storage with PostgreSQL and vector embeddings
- Semantic search for retrieving relevant memories
- Automatic memory deduplication
- In-memory store examples for development
- Agent middleware for memory loading and storage

## Structure

- `Agent/` - Main agent implementation with memory middleware
- `PostgresStore/` - PostgreSQL-based persistent storage setup
- `inMemoryStore/` - In-memory storage examples
- `demo/` - Basic usage examples
