from crewai import Crew, Process
from agents.ppt_agents import create_content_planner_agent, create_slide_designer_agent
from tasks.ppt_tasks import create_content_planning_task, create_slide_design_task

def create_ppt_presentation_crew(presentation_topic, target_audience):
    """
    Creates a crew of agents that work together to create a PowerPoint presentation.
    
    Args:
        presentation_topic (str): The topic of the presentation
        target_audience (str): Who the presentation is intended for
        
    Returns:
        Crew: A configured crew ready to create a presentation
    """
    # Create the agents
    content_planner = create_content_planner_agent()
    slide_designer = create_slide_designer_agent()
    
    # Create the tasks
    planning_task = create_content_planning_task(
        content_planner, 
        presentation_topic, 
        target_audience
    )
    design_task = create_slide_design_task(
        slide_designer,
        planning_task
    )
    
    # Create and return the crew
    return Crew(
        agents=[content_planner, slide_designer],
        tasks=[planning_task, design_task],
        verbose=True,
        process=Process.sequential  # Tasks are performed in sequence
    )
