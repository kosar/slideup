from crewai import Agent
from langchain_openai import ChatOpenAI
from config import DEFAULT_LLM_MODEL, TEMPERATURE, OPENAI_API_KEY

def create_content_planner_agent():
    """
    Creates an agent responsible for planning the PowerPoint content.
    This agent determines the overall structure and content strategy.
    """
    return Agent(
        role='PowerPoint Content Strategist',
        goal='Create well-structured content plans for presentations',
        backstory="""You are an expert presentation strategist with years of experience
        creating compelling PowerPoint decks for executives. You understand how to 
        structure information for maximum impact and clarity.""",
        verbose=True,
        llm=ChatOpenAI(
            model=DEFAULT_LLM_MODEL,
            temperature=TEMPERATURE,
            api_key=OPENAI_API_KEY
        ),
    )

def create_slide_designer_agent():
    """
    Creates an agent responsible for designing individual slides.
    This agent focuses on visual layout and design principles.
    """
    return Agent(
        role='Slide Design Specialist',
        goal='Design visually appealing and effective slides',
        backstory="""You are a talented presentation designer who knows how to
        create beautiful, clear, and impactful slides. You understand design principles,
        visual hierarchy, and how to effectively present information visually.""",
        verbose=True,
        llm=ChatOpenAI(
            model=DEFAULT_LLM_MODEL,
            temperature=TEMPERATURE,
            api_key=OPENAI_API_KEY
        ),
    )

# Additional specialized agents can be added here
