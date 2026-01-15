from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent, AgentState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import dynamic_prompt ,before_agent, after_agent
from langgraph.store.base import BaseStore
from langgraph.store.postgres import PostgresStore
from langchain.embeddings import init_embeddings
from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate
import uuid
from langgraph.runtime import Runtime

load_dotenv()

# ============= SETUP POSTGRES STORE =============
# Initialize PostgreSQL store with vector search
conn_string = "postgresql://postgres:postgres@localhost:5699/postgres?sslmode=disable"

# Keep the context manager alive by storing it
_store_ctx = PostgresStore.from_conn_string(
    conn_string,
    index={
        "dims": 1536,
        "embed": init_embeddings("openai:text-embedding-3-small"),
        "fields": ["data"]
    }
)

# PostgresStore.from_conn_string() returns a context manager
# We call __enter__() to get the actual PostgresStore object
# and keep the context manager alive for the entire program lifecycle
store: BaseStore = _store_ctx.__enter__()

try:
    store.setup()
    print("PostgresStore setup complete")
except Exception as e:
    print(f"Setup already done or error: {e}")

# ============= DATA MODELS =============
@dataclass
class Context:
    user_name: str
    store: PostgresStore
    memories: List[str] | None = None

class MemoryItem(BaseModel):
    text: str = Field(description="Atomic user memory")
    is_new: bool = Field(description="True if new, false if duplicate")

class MemoryDecision(BaseModel):
    should_write: bool
    memories: List[MemoryItem] = Field(default_factory=list)

# ============= LLM SETUP =============
llm = init_chat_model("gpt-4o")
memory_decide_llm = llm.with_structured_output(MemoryDecision)

# ============= PROMPTS =============
SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant with memory capabilities.

Your goal is to provide relevant, friendly, and tailored assistance that reflects the user's preferences, context, and past interactions.

If the user's name or relevant personal context is available, always personalize your responses by:
    – Always address the user by name (e.g., "Sure, Aayushmaan...") when appropriate
    – Reference known projects, tools, or preferences
    – Adjust the tone to feel friendly and direct

Avoid generic phrasing when personalization is possible."""

MEMORY_PROMPT = """Analyze if this user message contains new, important memories to store.
User: {user_name}
Existing memories: {memories}

Should you store new memories? Return structured decision."""

# ============= DYNAMIC PROMPT MIDDLEWARE =============
@dynamic_prompt
def change_prompt(request):
    system_prompt = SYSTEM_PROMPT_TEMPLATE
    
    memories = request.runtime.context.memories
    if memories:
        memories_str = "\n".join([f"- {m}" for m in memories])
        system_prompt += f"\n\nUser's relevant memories:\n{memories_str}"
        print("Adding memories to system prompt")
    
    return system_prompt

# ============= BEFORE AGENT MIDDLEWARE =============
@before_agent
def load_messages(state: AgentState, runtime: Runtime):
    print("Loading memories...")
    
    ctx = runtime.context
    if not ctx or not isinstance(ctx, Context):
        print("Warning: No valid context")
        return None
    
    user_name = ctx.user_name
    store = ctx.store
    
    # Use namespace tuple format for PostgresStore
    namespace = ("users", user_name, "details")
    last_msg = state["messages"][-1].content
    
    try:
        # Search in PostgresStore using semantic search
        items = store.search(namespace, query=last_msg, limit=5)
        
        if items:
            # Extract memory text from search results
            ctx.memories = [item.value.get("data", "") for item in items]
            print(f"Found {len(items)} relevant memories")
        else:
            ctx.memories = []
            print("No memories found")
    except Exception as e:
        print(f"Error loading memories: {e}")
        ctx.memories = []
    
    return None

# ============= AFTER AGENT MIDDLEWARE =============
@after_agent
def store_messages(state: AgentState, runtime: Runtime):
    print("Storing memories...")
    
    ctx = runtime.context
    if not ctx or not isinstance(ctx, Context):
        print("Warning: No valid context")
        return None
    
    user_name = ctx.user_name
    store = ctx.store
    
    namespace = ("users", user_name, "details")
    
    try:
        # Get existing memories from PostgresStore
        existing_items = store.search(namespace, limit=10)
        existing = "\n".join(item.value.get("data", "") for item in existing_items) if existing_items else "(no memories)"
        
        # Get the user's message (second to last in state)
        last_msg = state["messages"][-2].content
        
        # Format the memory decision prompt
        memory_prompt_text = MEMORY_PROMPT.format(
            user_name=user_name,
            memories=existing
        )
        
        # Create prompt for LLM
        prompt = ChatPromptTemplate.from_messages([
            ("system", memory_prompt_text),
            ("human", last_msg),
        ])
        
        # Get memory decision from LLM
        decision = memory_decide_llm.invoke(prompt.format_prompt().to_messages())
        
        if decision.should_write:
            for mem in decision.memories:
                if mem.is_new and mem.text.strip():
                    # Store in PostgresStore
                    # namespace: tuple, key: unique id, value: dict with data field
                    store.put(
                        namespace,
                        str(uuid.uuid4()),
                        {"data": mem.text.strip()}
                    )
            new_count = len([m for m in decision.memories if m.is_new])
            print(f"Stored {new_count} new memories")
        else:
            print("Skipping memory storage")
    except Exception as e:
        print(f"Error storing memories: {e}")
    
    return None

# ============= AGENT CREATION =============
agent = create_agent(
    model=llm,
    tools=[],
    system_prompt=SYSTEM_PROMPT_TEMPLATE,
    checkpointer=InMemorySaver(),
    context_schema=Context,
    middleware=[load_messages, store_messages, change_prompt]
    # Note: create_agent doesn't have a store parameter,
    # so you must pass it in context during invoke()
)

# Same thread_id persists history across calls
config = {"configurable": {"thread_id": "chat-1"}}

# ============= TEST INVOCATION =============
if __name__ == "__main__":
    print("Welcome to Long Term Memory Test for Postgres Version")
    name = input("Please enter your name: ")
    
    print("\n" + "=" * 60)
    print("TEST 1: First message")
    print("=" * 60)
    result1 = agent.invoke({
        "messages": [HumanMessage(content="Hi, my name is Aayushmaan. I'm working on a RAG evaluation system.")]
    }, config, context=Context(user_name=name, store=store))
    
    print("\nAgent response:")
    print(result1["messages"][-1].content)
    
    print("\n" + "=" * 60)
    print("TEST 2: Follow-up question (should retrieve memory)")
    print("=" * 60)
    result2 = agent.invoke({
        "messages": [HumanMessage(content="What frameworks should I use?")]
    }, config, context=Context(user_name=name, store=store))
    
    print("\nAgent response:")
    print(result2["messages"][-1].content)


    print("=" * 60)
    print("TEST 3: Related question - Should retrieve RAG memory")
    print("=" * 60)
    result3 = agent.invoke({
        "messages": [HumanMessage(content="How do I evaluate RAG systems?")]
    }, config, context=Context(user_name=name, store=store))

    print("\nAgent response:")
    print(result3["messages"][-1].content)
    print("\n")


    print("=" * 60)
    print("TEST 4: New context - Add more memories")
    print("=" * 60)
    result4 = agent.invoke({
        "messages": [HumanMessage(content="I also use LangGraph for multi-agent systems and I'm preparing for AI engineering interviews.")]
    }, config, context=Context(user_name=name, store=store))

    print("\nAgent response:")
    print(result4["messages"][-1].content)
    print("\n")


    print("=" * 60)
    print("TEST 5: Related to new memory - Should use LangGraph memory")
    print("=" * 60)
    result5 = agent.invoke({
        "messages": [HumanMessage(content="What are best practices for LangGraph?")]
    }, config, context=Context(user_name=name, store=store))

    print("\nAgent response:")
    print(result5["messages"][-1].content)
    print("\n")


    print("=" * 60)
    print("TEST 6: Unrelated question - Minimal memory retrieval")
    print("=" * 60)
    result6 = agent.invoke({
        "messages": [HumanMessage(content="What's the weather like?")]
    }, config, context=Context(user_name=name, store=store))

    print("\nAgent response:")
    print(result6["messages"][-1].content)
    print("\n")


    print("=" * 60)
    print("TEST 7: Interview prep question - Should use interview memory")
    print("=" * 60)
    result7 = agent.invoke({
        "messages": [HumanMessage(content="How should I prepare for AI engineering interviews?")]
    }, config, context=Context(user_name=name, store=store))

    print("\nAgent response:")
    print(result7["messages"][-1].content)
    print("\n")


    print("=" * 60)
    print("TEST 8: Different user - No shared memories")
    print("=" * 60)
    other_name = "John"
    result8 = agent.invoke({
        "messages": [HumanMessage(content="Hi, I'm working on web development")]
    }, {"configurable": {"thread_id": "chat-2"}}, context=Context(user_name=other_name, store=store))

    print("\nAgent response:")
    print(result8["messages"][-1].content)
    print("\n")


    print("=" * 60)
    print("TEST 9: Back to original user - Should still have old memories")
    print("=" * 60)
    result9 = agent.invoke({
        "messages": [HumanMessage(content="Remind me what I'm working on")]
    }, config, context=Context(user_name=name, store=store))

    print("\nAgent response:")
    print(result9["messages"][-1].content)
    print("\n")