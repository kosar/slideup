from crewai import Agent, Task, Crew, Process
from config import validate_config, DEFAULT_LLM_MODEL

def main():
    # Validate configuration
    validate_config()
    
    # This will be your entry point to create and run crews
    print("PowerPoint Agentic System Initialized")
    print(f"Using model: {DEFAULT_LLM_MODEL}")
    
    # TODO: Import and initialize your agents, tasks and crews here
    
    # Example crew setup (to be implemented)
    # ppt_crew = create_ppt_crew()
    # result = ppt_crew.kickoff()
    # print(result)

if __name__ == "__main__":
    main()
