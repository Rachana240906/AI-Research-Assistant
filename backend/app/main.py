from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Explicitly load the .env environment variables before importing our service
load_dotenv()

# Now we can safely import the service since the environment variable is loaded
from app.services.agent import ResearchAgentService

app = FastAPI(title="AI Research Assistant API")

# Configure CORS Middleware so our pink frontend can talk to it safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local execution
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Initialize our working agent service 
agent_service = ResearchAgentService()

class ResearchRequest(BaseModel):
    query: str

class ResearchResponse(BaseModel):
    query: str
    output: str

@app.post("/api/research", response_model=ResearchResponse)
async def research_endpoint(request: ResearchRequest):
    # Run our decoupled lookup process
    agent_output = agent_service.run_research(request.query)
    
    return ResearchResponse(
        query=request.query,
        output=agent_output
    )

@app.get("/")
def read_root():
    return {"status": "Backend running cleanly with CORS enabled!"}