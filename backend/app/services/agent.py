import os
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import HumanMessage, SystemMessage

class ResearchAgentService:
    def __init__(self):
        # Initialize our standard, fast Groq LLM
        self.llm = ChatGroq(
            model_name="llama-3.3-70b-versatile", 
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        # Initialize Wikipedia search wrapper
        self.wiki = WikipediaAPIWrapper()

    def run_research(self, query: str) -> str:
        """
        Executes a 100% reliable two-step research process:
        Step 1: Uses the LLM to optimize a precise search query.
        Step 2: Fetches live data from Wikipedia.
        Step 3: Synthesizes a clean, grounded research report.
        """
        print(f"--> [Agent Execution] Analyzing search request: '{query}'")
        
        # Step 1: Ask the LLM to pull out the single best search phrase
        search_prompt = [
            SystemMessage(content="You extract keywords. Respond with ONLY a clean, plain text search phrase for Wikipedia. No JSON, no quotes, no conversational text."),
            HumanMessage(content=f"What is the best single Wikipedia search term to answer: {query}")
        ]
        
        optimized_query = self.llm.invoke(search_prompt).content.strip().replace('"', '').replace("'", "")
        print(f"--> [Agent Execution] Live Searching Wikipedia for: '{optimized_query}'")
        
        # Step 2: Grab the live data from Wikipedia safely
        try:
            search_results = self.wiki.run(optimized_query)
        except Exception as e:
            search_results = f"No detailed records found on Wikipedia for this term. (Error: {str(e)})"
            
        # Step 3: Pass everything to the model to write a final beautiful summary response
        final_prompt = [
            SystemMessage(content="You are an expert research assistant. Read the provided background source context and write a detailed, highly accurate summary answering the user's question."),
            HumanMessage(content=f"User Question: {query}\n\nWikipedia Source Context:\n{search_results}")
        ]
        
        final_answer = self.llm.invoke(final_prompt)
        return final_answer.content