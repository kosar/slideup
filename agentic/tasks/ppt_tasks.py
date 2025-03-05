from crewai import Task

def create_content_planning_task(agent, topic, audience):
    """
    Creates a task for planning the content of a presentation.
    
    Args:
        agent: The agent who will perform this task
        topic (str): The presentation topic
        audience (str): The target audience
        
    Returns:
        Task: A configured content planning task
    """
    return Task(
        description=f"""
        Plan a compelling PowerPoint presentation on the topic: "{topic}"
        for an audience of {audience}.
        
        The plan should include:
        1. Overall presentation structure (beginning, middle, end)
        2. Key messages to convey
        3. Number of slides recommended
        4. Types of content for each slide (text, images, charts, etc.)
        5. Narrative flow between slides
        
        Be specific and strategic in your planning to create maximum impact.
        """,
        agent=agent,
    )

def create_slide_design_task(agent, planning_task):
    """
    Creates a task for designing the slides based on the content plan.
    
    Args:
        agent: The agent who will perform this task
        planning_task: The previous planning task that provides the content plan
        
    Returns:
        Task: A configured slide design task
    """
    return Task(
        description=f"""
        Based on the content plan provided, design the individual slides
        for the presentation. For each slide, provide:
        
        1. Slide layout recommendation
        2. Content placement
        3. Visual elements to include
        4. Color scheme suggestions
        5. Specific design notes to enhance the message
        
        Focus on creating a cohesive, professional, and impactful design
        that effectively communicates the key messages.
        """,
        agent=agent,
        context=[planning_task]
    )
